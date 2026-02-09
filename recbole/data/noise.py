import torch
import numpy as np


def get_long_tail_index(dataset, k=0.2):
    """
    根据物品的交互频率将物品分为头部（head）和尾部（tail）
    
    Args:
        dataset: RecBole 数据集对象
        k: 头部物品的比例，默认0.2表示前20%最流行的物品为头部
    
    Returns:
        head: 头部物品的ID张量（高频物品）
        tail: 尾部物品的ID张量（低频物品）
    """
    # 获取所有交互数据
    inter_feat = dataset.inter_feat
    item_ids = inter_feat.interaction['item_id']
    
    # 统计每个物品的交互次数（流行度）
    item_counts = {}
    for item_id in item_ids.tolist():
        if item_id not in item_counts:
            item_counts[item_id] = 0
        item_counts[item_id] += 1
    
    # 按交互次数降序排序
    sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
    
    # 计算头部物品的数量
    n_head = int(len(sorted_items) * k)
    
    # 分割头部和尾部
    head_items = [item_id for item_id, _ in sorted_items[:n_head]]
    tail_items = [item_id for item_id, _ in sorted_items[n_head:]]
    
    # 转换为 PyTorch 张量
    head = torch.tensor(head_items, dtype=torch.long)
    tail = torch.tensor(tail_items, dtype=torch.long)
    
    print(f"[Long-tail Split] Total items: {len(sorted_items)}, "
          f"Head items (top {k*100:.0f}%): {len(head_items)}, "
          f"Tail items: {len(tail_items)}")
    
    return head, tail


def feast(valid_data, test_data, dataset, k=0.2):
    """
    对验证集和测试集进行长尾分布过滤
    
    Args:
        valid_data: 验证集数据
        test_data: 测试集数据
        dataset: 原始数据集
        k: 头部物品比例，默认0.2表示前20%为头部
    
    Returns:
        处理后的 valid_data, test_data
    """
    from recbole.data.interaction import Interaction
    
    valid_data_inter_dict = valid_data.inter_feat.interaction
    
    # 获取头部（head）和尾部（tail）物品的索引
    head, tail = get_long_tail_index(dataset, k=k)
    
    # -------------------------------------------------------
    # 处理验证集 (修改为：也保留 Tail 物品，与测试集保持一致)
    # -------------------------------------------------------
    
    # 步骤 1: 生成布尔掩码 (筛选属于 tail 的物品)
    mask = torch.isin(valid_data_inter_dict['item_id'], tail)
    
    # 步骤 2: 对所有字段应用掩码过滤
    filtered_interaction = {}
    for key, value in valid_data_inter_dict.items():
        filtered_interaction[key] = value[mask]
    
    # 关键修改：重新实例化 Interaction 对象，以便自动更新长度
    valid_data.inter_feat = Interaction(filtered_interaction)
    
    print(f"[Long-tail Split] Validation set is now filtered to keep TAIL items (aligned with Test set).")
    
    # -------------------------------------------------------
    # 处理测试集 (保留 Tail 物品)
    # -------------------------------------------------------
    
    test_data_inter_dict = test_data.inter_feat.interaction
    
    # 步骤 1: 生成布尔掩码 (筛选属于 tail 的物品)
    mask = torch.isin(test_data_inter_dict['item_id'], tail)
    
    # 步骤 2: 对所有字段应用掩码过滤
    filtered_interaction = {}
    for key, value in test_data_inter_dict.items():
        filtered_interaction[key] = value[mask]
    
    # 关键修改：重新实例化 Interaction 对象，以便自动更新长度
    test_data.inter_feat = Interaction(filtered_interaction)
    
    return valid_data, test_data


def add_noise(valid_data, test_data, dataset, noise_ratio=0.05):
    """
    对验证集和测试集注入噪声（标签错误）
    
    Args:
        valid_data: 验证集数据
        test_data: 测试集数据
        dataset: 原始数据集（用于获取物品范围）
        noise_ratio: 噪声比例，默认5%
    
    Returns:
        处理后的 valid_data, test_data
    """
    import torch
    import numpy as np
    
    # 处理验证集
    valid_data = inject_noise_to_dataset(valid_data, dataset, noise_ratio)
    
    # 处理测试集
    test_data = inject_noise_to_dataset(test_data, dataset, noise_ratio)
    
    return valid_data, test_data


def inject_noise_to_dataset(data, dataset, noise_ratio):
    """
    对单个数据集注入标签噪声
    
    实现方式：随机选择 noise_ratio 比例的交互记录，将其 item_id 替换成随机的其他 item_id
    
    Args:
        data: 数据集（验证集或测试集）
        dataset: 原始数据集（用于获取物品ID范围）
        noise_ratio: 噪声比例
    
    Returns:
        注入噪声后的数据集
    """
    import torch
    import numpy as np
    
    data_inter = data.inter_feat.interaction
    
    user_ids = data_inter['user_id']
    item_ids = data_inter['item_id']
    
    # 获取所有可能的物品ID（从1到item_num-1，因为0通常是padding）
    all_items = torch.arange(1, dataset.item_num)
    
    # 计算需要加噪声的样本数量
    n_samples = len(user_ids)
    n_noise = int(n_samples * noise_ratio)
    
    print(f"[Noise Injection] Total samples: {n_samples}, Injecting noise to {n_noise} samples ({noise_ratio*100:.1f}%)")
    
    # 随机选择需要加噪声的索引
    np.random.seed(42)  # 设置随机种子以保证可重复性
    noise_indices = np.random.choice(n_samples, n_noise, replace=False)
    
    # 复制item_ids以便修改（避免直接修改原数据）
    noisy_item_ids = item_ids.clone()
    
    # 对选中的样本，随机替换item_id为错误的标签
    for idx in noise_indices:
        original_item = item_ids[idx].item()
        
        # 从所有物品中随机选择一个不同的item_id（模拟标签错误）
        random_item = all_items[torch.randint(0, len(all_items), (1,))].item()
        
        # 确保选择的item与原item不同
        max_attempts = 10
        attempts = 0
        while random_item == original_item and attempts < max_attempts:
            random_item = all_items[torch.randint(0, len(all_items), (1,))].item()
            attempts += 1
        
        noisy_item_ids[idx] = random_item
    
    # 更新交互数据：直接修改 item_id 字段，保留所有其他字段
    data_inter['item_id'] = noisy_item_ids
    
    return data