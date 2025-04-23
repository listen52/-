import numpy as np
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50  # 使用ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input  # ResNet50的预处理函数
from tensorflow.keras.models import Model
import time  # 用于性能测量
import uuid  # 用于生成唯一ID
import json  # 用于JSON序列化
from qdrant_client import QdrantClient  # 导入Qdrant客户端
from qdrant_client.http import models  # 导入Qdrant模型定义
import tensorflow as tf
print("TensorFlow 版本:", tf.__version__)
print("可用的 GPU 设备:", tf.config.list_physical_devices('GPU'))

# 仅在需要时分配 GPU 内存
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 设置 TensorFlow 仅在需要时分配 GPU 内存
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("已启用按需分配 GPU 内存")
    except RuntimeError as e:
        print(e)
    
# --- 1. 加载预训练模型并构建特征提取器 ---
# 使用ResNet50，去掉顶部分类层，添加全局平均池化
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

# ResNet50的标准输入尺寸为(224, 224)
input_shape = (224, 224)

def preprocess_image(img_path, target_size):
    """加载并预处理图片以适配模型输入"""
    try:
        img = Image.open(img_path).convert('RGB')  # 确保是RGB格式
        img = img.resize(target_size)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)  # 增加batch维度
        img_array = preprocess_input(img_array)  # 使用ResNet50特定的预处理函数
        return img_array
    except Exception as e:
        print(f"处理图片时出错 {img_path}: {e}")
        return None

def extract_features(img_path, model, target_size):
    """提取单张图片的特征向量"""
    preprocessed_img = preprocess_image(img_path, target_size)
    if preprocessed_img is not None:
        features = model.predict(preprocessed_img)
        return features.flatten()  # 展平成一维向量
    return None

# --- 2. 为图片库建立特征向量索引 ---
def compute_features_for_database(image_folder, model, target_size):
    """计算文件夹中所有图片的特征向量"""
    features_db = {}
    count = 0
    total_files = len([f for f in os.listdir(image_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(image_folder, filename)
            features = extract_features(img_path, model, target_size)
            if features is not None:
                features_db[img_path] = features
                count += 1
                if count % 10 == 0:  # 每处理10张图片显示一次进度
                    print(f"处理进度: {count}/{total_files} ({count/total_files*100:.1f}%)")
            else:
                print(f"无法提取特征: {filename}")
    return features_db

# --- 3. 向量标准化函数 ---
def normalize_vectors(vectors):
    """对向量进行L2归一化，使其可以通过内积计算余弦相似度"""
    # 计算每个向量的L2范数
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # 避免除以0
    norms = np.maximum(norms, 1e-15)
    # 归一化向量
    normalized_vectors = vectors / norms
    return normalized_vectors

# --- 4. Qdrant索引相关函数 ---
def create_qdrant_collection(client, collection_name, vector_size):
    """创建Qdrant集合"""
    print(f"正在创建Qdrant集合: {collection_name}")
    
    # 检查集合是否已存在
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    
    if collection_name in collection_names:
        print(f"集合 {collection_name} 已存在，将使用现有集合")
        return True
    
    # 创建集合，使用余弦相似度（等同于内积，当向量被归一化时）
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE  # 使用余弦相似度
        ),
        hnsw_config=models.HnswConfigDiff(
            m=16,  # 每个节点的最大邻居数
            ef_construct=100  # 构建时的探索因子
        )
    )
    print(f"已创建Qdrant集合: {collection_name}")
    return True

def add_vectors_to_qdrant(client, collection_name, paths, vectors):
    """将向量添加到Qdrant集合"""
    print(f"正在将 {len(vectors)} 个向量添加到Qdrant集合...")
    
    # 首先归一化向量
    normalized_vectors = normalize_vectors(vectors)
    
    # 准备批量添加的点
    points = []
    path_map = {}  # 用于存储ID到路径的映射
    
    for idx, (path, vector) in enumerate(zip(paths, normalized_vectors)):
        point_id = idx  # 使用索引作为ID
        points.append(models.PointStruct(
            id=point_id,
            vector=vector.tolist(),
            payload={"image_path": path}  # 在payload中存储路径
        ))
        path_map[point_id] = path
    
    # 批量上传点
    client.upsert(
        collection_name=collection_name,
        points=points
    )
    print(f"已添加 {len(points)} 个向量到Qdrant集合")
    
    # 保存ID到路径的映射，用于后续搜索
    save_path_map(path_map)
    
    return True

def save_path_map(path_map, filename="qdrant_path_map.json"):
    """保存ID到路径的映射"""
    # 将整数键转换为字符串，因为JSON不支持整数键
    serializable_map = {str(k): v for k, v in path_map.items()}
    with open(filename, 'w') as f:
        json.dump(serializable_map, f)
    print(f"ID到路径映射已保存到 {filename}")

def load_path_map(filename="qdrant_path_map.json"):
    """加载ID到路径的映射"""
    try:
        with open(filename, 'r') as f:
            serialized_map = json.load(f)
        # 将字符串键转换回整数
        path_map = {int(k): v for k, v in serialized_map.items()}
        print(f"已从 {filename} 加载ID到路径映射")
        return path_map
    except Exception as e:
        print(f"加载ID到路径映射时出错: {e}")
        return None

# --- 5. 查询相似图片 ---
def find_similar_images_qdrant(query_image_path, client, collection_name, model, target_size, top_n=5):
    """使用Qdrant查找相似图片"""
    print(f"\n正在搜索与图片相似的图像: {query_image_path}")
    query_features = extract_features(query_image_path, model, target_size)

    if query_features is None:
        print("提取查询图片特征时出错。")
        return []
    
    # 归一化查询向量
    query_features = normalize_vectors(query_features.reshape(1, -1))
    
    # 搜索
    start_time = time.time()
    search_response = client.query_points(
        collection_name=collection_name,
        query=query_features[0].tolist(),
        limit=top_n,
        with_payload=True
    )
    end_time = time.time()
    
    print(f"Qdrant搜索耗时: {end_time - start_time:.4f}秒")
    
    # 从返回的QueryResponse中获取points
    points = search_response.points
    scores = [point.score for point in points]
    print(f"原始相似度分数: {scores}")
    
    # 整理结果
    results = []
    for point in points:
        image_path = point.payload.get("image_path")
        if image_path:
            # 在Qdrant中，余弦相似度越高表示越相似
            results.append((image_path, point.score))
    
    return results

# --- 6. 批量查询多张图片 ---
def batch_similar_search_qdrant(query_folder, client, collection_name, model, target_size, top_n=3):
    """批量查询多张图片的相似图片"""
    results = {}
    for filename in os.listdir(query_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            query_path = os.path.join(query_folder, filename)
            similar_images = find_similar_images_qdrant(query_path, client, collection_name, model, target_size, top_n)
            if similar_images:
                results[query_path] = similar_images
    return results

# --- 7. 主执行代码 ---
def main():
    # 配置参数
    database_path = r'D:\产品图片1'  # 图片数据库路径
    collection_name = "product_images"  # Qdrant集合名称
    path_map_file = "qdrant_path_map.json"  # ID到路径的映射文件
    query_path = r'D:\产品图片\124.jpg'  # 查询图片路径
    top_results = 5  # 返回结果数量
    
    # 连接到Qdrant服务器
    client = QdrantClient(host="localhost", port=6333)
    
    # 检查是否存在已创建的集合
    collections = client.get_collections().collections
    collection_exists = any(collection.name == collection_name for collection in collections)
    
    if collection_exists and os.path.exists(path_map_file):
        # 使用现有的集合
        print(f"使用现有的Qdrant集合: {collection_name}")
        path_map = load_path_map(path_map_file)
        
        if path_map is None:
            # 加载映射失败，重新处理
            print("加载ID到路径映射失败，正在重新处理...")
            database_features = compute_features_for_database(database_path, model, input_shape)
            db_paths = list(database_features.keys())
            db_vectors = np.array(list(database_features.values()))
            
            # 创建集合并添加向量
            create_qdrant_collection(client, collection_name, db_vectors.shape[1])
            add_vectors_to_qdrant(client, collection_name, db_paths, db_vectors)
    else:
        # 没有找到已创建的集合或映射文件，计算特征向量并创建索引
        print("未找到已创建的集合或映射文件，正在重新计算...")
        print("正在使用ResNet50构建特征数据库...")
        database_features = compute_features_for_database(database_path, model, input_shape)
        print(f"已为{len(database_features)}张图片创建特征向量。")
        
        db_paths = list(database_features.keys())
        db_vectors = np.array(list(database_features.values()))
        
        # 创建集合并添加向量
        vector_size = db_vectors.shape[1]
        create_qdrant_collection(client, collection_name, vector_size)
        add_vectors_to_qdrant(client, collection_name, db_paths, db_vectors)
    
    # 执行搜索
    similar_results = find_similar_images_qdrant(query_path, client, collection_name, model, input_shape, top_n=top_results)
    
    # 显示结果
    if similar_results:
        print(f"\n找到前{len(similar_results)}个相似图片:")
        for img_path, score in similar_results:
            # 排除查询图片本身（如果它也在数据库中）
            if os.path.abspath(img_path) != os.path.abspath(query_path):
                 print(f"- {img_path} (相似度分数: {score:.4f})")
    else:
        print("\n未找到相似图片。")

# --- 8. 执行入口 ---
if __name__ == "__main__":
    main() 