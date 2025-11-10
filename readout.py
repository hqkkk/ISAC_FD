import os
import torch
import numpy as np
import pandas as pd


def describe_tensor(name, tensor, max_items=6):
    """打印张量的简短描述和预览（支持复数）。"""
    if tensor is None:
        print(f"{name}: None")
        return

    t = tensor
    # 如果是 torch.Tensor，转换为 CPU 上的 numpy
    if isinstance(t, torch.Tensor):
        t_cpu = t.detach().cpu()
        dtype = str(t_cpu.dtype)
        shape = tuple(t_cpu.shape)
        np_arr = t_cpu.numpy()
    else:
        # 已经是 numpy
        np_arr = np.array(t)
        dtype = str(np_arr.dtype)
        shape = np_arr.shape

    print(f"{name} → dtype: {dtype}, shape: {shape}")

    # 预览内容
    # 对于标量或小数组直接打印，对较大数组打印前几个元素/行
    np.set_printoptions(precision=4, suppress=True)
    try:
        if np_arr.size == 0:
            print("  (empty)")
            return

        # 如果是复数数组，展示实部/虚部
        if np.iscomplexobj(np_arr):
            flat = np_arr.flatten()
            n = min(max_items, flat.size)
            sample = flat[:n]
            sample_str = ", ".join([f"{z.real:.4f}+{z.imag:.4f}j" for z in sample])
            print(f"  preview (first {n} elements, real+imag): [{sample_str}]")
        else:
            # 实数数组
            flat = np_arr.flatten()
            n = min(max_items, flat.size)
            sample = flat[:n]
            sample_str = ", ".join([f"{float(x):.4f}" for x in sample])
            print(f"  preview (first {n} elements): [{sample_str}]")
    except Exception as e:
        print(f"  (could not preview: {e})")


def load_dataset(data_dir="dataset", dataset_name="dataset.pt", info_name="data_info.pt"):
    dataset_path = os.path.join(data_dir, dataset_name)
    info_path = os.path.join(data_dir, info_name)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"找不到数据文件: {dataset_path}")
    if not os.path.exists(info_path):
        print(f"警告: 找不到信息文件 {info_path}，将只读取数据文件")

    print(f"加载数据: {dataset_path}")
    # 在 PyTorch 2.6+ 中, torch.load 默认使用 weights_only=True 增强安全性，
    # 这会阻止反序列化某些非安全类型（例如 numpy 的 scalar）。
    # 我们尝试安全恢复：先使用 add_safe_globals 允许 numpy scalar，再回退到 weights_only=False。
    try:
        data = torch.load(dataset_path, map_location="cpu")
    except Exception as e:
        # 捕获 UnpicklingError 并尝试允许 numpy scalar
        print(f"首次加载失败: {e}")
        try:
            # 尝试将 numpy scalar 加入 allowlist（需信任数据来源）
            if hasattr(torch, 'serialization') and hasattr(torch.serialization, 'add_safe_globals'):
                try:
                    torch.serialization.add_safe_globals([np._core.multiarray.scalar])
                    print("已将 numpy._core.multiarray.scalar 加入反序列化安全白名单，重试加载...")
                    data = torch.load(dataset_path, map_location="cpu")
                except Exception as e2:
                    print(f"通过 add_safe_globals 重试失败: {e2}")
                    raise
            else:
                raise
        except Exception:
            # 最终回退（较不安全）——显式允许完整反序列化
            print("回退到 torch.load(..., weights_only=False) 以完成加载（仅在信任数据时使用）。")
            data = torch.load(dataset_path, map_location="cpu", weights_only=False)

    info = None
    if os.path.exists(info_path):
        print(f"加载数据描述: {info_path}")
        try:
            info = torch.load(info_path, map_location="cpu")
        except Exception:
            # 同样处理 info 加载中的安全限制——尝试安全白名单后回退
            try:
                if hasattr(torch, 'serialization') and hasattr(torch.serialization, 'add_safe_globals'):
                    torch.serialization.add_safe_globals([np._core.multiarray.scalar])
                    info = torch.load(info_path, map_location="cpu")
                else:
                    info = torch.load(info_path, map_location="cpu", weights_only=False)
            except Exception as e:
                print(f"加载 data_info 时遇到错误: {e}")
                info = None

    return data, info


def tensor_to_dataframes(name, np_arr):
    """将 numpy 数组转换为一个或多个 (title, DataFrame) 对，用于写入 Excel。
    复数数组会被拆分为实部和虚部两个表格。
    """
    np_arr = np.array(np_arr)
    dfs = []
    if np_arr.dtype.kind == 'c':
        real = np_arr.real
        imag = np_arr.imag
        dfs.append((f"{name} (real)", pd.DataFrame(real)))
        dfs.append((f"{name} (imag)", pd.DataFrame(imag)))
        return dfs

    # 实数情况，根据维度处理
    if np_arr.ndim == 0:
        dfs.append((name, pd.DataFrame([[np_arr.item()]])))
    elif np_arr.ndim == 1:
        dfs.append((name, pd.DataFrame(np_arr.reshape(1, -1))))
    elif np_arr.ndim == 2:
        dfs.append((name, pd.DataFrame(np_arr)))
    else:
        # 高维张量：压平除第一维外的维度，或整体压平
        try:
            reshaped = np_arr.reshape(np_arr.shape[0], -1)
            dfs.append((name + " (reshaped)", pd.DataFrame(reshaped)))
        except Exception:
            dfs.append((name + " (flatten)", pd.DataFrame(np_arr.flatten()).T))
    return dfs


def save_samples_to_excel(data, info, num_samples, n_show, desc_map, out_path):
    """将前 n_show 个样本完整写入 Excel，每个样本一个 sheet，包含提示信息。"""
    dirpath = os.path.dirname(out_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    with pd.ExcelWriter(out_path, engine='xlsxwriter') as writer:
        for i in range(n_show):
            sheet = f"Sample_{i+1}"
            startrow = 0

            # 写入样本标题
            title = f"样本 {i+1} (index {i}) — 完整内容"
            pd.DataFrame([title]).to_excel(writer, sheet_name=sheet, startrow=startrow, index=False, header=False)
            startrow += 2

            # 写入 info 元数据（如果有）
            if info is not None and isinstance(info, dict):
                info_df = pd.DataFrame(list(info.items()), columns=['key', 'value'])
                info_df.to_excel(writer, sheet_name=sheet, startrow=startrow, index=False)
                startrow += len(info_df) + 2

            # 逐键写入样本数据
            for k in data.keys():
                v = data[k]
                # 获取样本级数据（如果第0维为样本维则取第 i 个）
                if isinstance(v, torch.Tensor):
                    if v.ndim >= 1 and v.shape[0] == num_samples:
                        sample_v = v[i].detach().cpu().numpy()
                    else:
                        sample_v = v.detach().cpu().numpy()
                else:
                    sample_v = v

                # 写入描述
                desc = desc_map.get(k, '')
                pd.DataFrame([f"{k}: {desc}"]).to_excel(writer, sheet_name=sheet, startrow=startrow, index=False, header=False)
                startrow += 1

                # 写入值（完整）
                try:
                    if isinstance(sample_v, (int, float, str)):
                        pd.DataFrame([sample_v], columns=[k]).to_excel(writer, sheet_name=sheet, startrow=startrow, index=False)
                        startrow += 2
                    else:
                        dfs = tensor_to_dataframes(k, sample_v)
                        for title, df in dfs:
                            pd.DataFrame([title]).to_excel(writer, sheet_name=sheet, startrow=startrow, index=False, header=False)
                            startrow += 1
                            df.to_excel(writer, sheet_name=sheet, startrow=startrow, index=True)
                            startrow += len(df) + 2
                except Exception as e:
                    pd.DataFrame([f"无法写入 {k}: {e}"]).to_excel(writer, sheet_name=sheet, startrow=startrow, index=False, header=False)
                    startrow += 1

    print(f"已将前 {n_show} 个样本完整保存到: {out_path}")


def main():
    print("=== readout.py: 从 data_create 保存的文件中读取前10个样本并显示说明 ===")
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "dataset")

    try:
        data, info = load_dataset(data_dir=data_dir)
    except FileNotFoundError as e:
        print(e)
        return

    # 打印 data_info（如果存在）
    if info is not None:
        print("\n--- data_info 内容（元数据）---")
        if isinstance(info, dict):
            for k, v in info.items():
                print(f"{k}: {v}")
        else:
            print(info)
    else:
        print("\n未找到 data_info（元数据）。")

    # data 应该是一个字典，包含 'UUMat','DUMat','INMat','TAMat','CIMat' 等
    if not isinstance(data, dict):
        print("数据文件格式不是字典，无法按预期读取。内容预览:")
        print(type(data))
        return

    expected_keys = ['UUMat', 'DUMat', 'INMat', 'TAMat', 'CIMat']
    present_keys = list(data.keys())
    print(f"\n数据文件包含键: {present_keys}")

    # 确定样本数量：尽量以 UUMat 的第0维为准
    num_samples = None
    for k in expected_keys:
        if k in data and isinstance(data[k], torch.Tensor):
            if data[k].ndim >= 1:
                num_samples = data[k].shape[0]
                break

    if num_samples is None:
        # 后备：从任意 tensor 的第一维推断
        for v in data.values():
            if isinstance(v, torch.Tensor) and v.ndim >= 1:
                num_samples = v.shape[0]
                break

    if num_samples is None:
        num_samples = 1

    print(f"检测到样本数量 (推断): {num_samples}")

    n_show = min(10, int(num_samples))
    print(f"\n将显示前 {n_show} 个样本的详细信息（若可用）\n")

    # 对每个样本，打印键的含义、shape、dtype 和预览
    desc_map = {
        'UUMat': 'UU到BS信道矩阵（每个样本：UUnum × N_r）——上行用户到基站接收天线的 CSI',
        'DUMat': 'BS到DU信道矩阵（每个样本：DUnum × N_t）——基站发射到下行用户的 CSI',
        'INMat': '干扰源参数矩阵（每个样本：INnum × 2）——列: [angle, powerGain]（以复数保存）',
        'TAMat': '目标参数矩阵（每个样本：TAnum × 2）——列: [angle, powerGain]（以复数保存）',
        'CIMat': 'UU与DU之间的互相干扰CSI矩阵（每个样本：UUnum × DUnum）'
    }

    for i in range(n_show):
        print(f"{'='*60}\n样本 {i+1} (index {i}):\n{'-'*60}")
        for k in present_keys:
            # 跳过不是 tensor 的值（例如 noise 参数），但也打印说明
            v = data[k]
            if isinstance(v, torch.Tensor):
                # 如果第0维是样本维，则取该样本，否则直接使用
                if v.ndim >= 1 and v.shape[0] == num_samples:
                    sample_v = v[i]
                else:
                    # 不是以样本为首维，直接当作全局项显示
                    sample_v = v
                print(f"\n{k}: {desc_map.get(k, '')}")
                describe_tensor(k, sample_v)
            else:
                # 不是 tensor 的，可能是噪声或标量参数
                if isinstance(v, (int, float, str)):
                    print(f"\n{k}: (标量) {v}")
                else:
                    print(f"\n{k}: (非张量类型，类型={type(v)})")
                    try:
                        print(v)
                    except:
                        pass

    print('\n完成：已打印所选样本的基本信息与预览。')
    # 将前 n_show 个样本保存为 Excel（完整内容）
    out_file = os.path.join(base_dir, 'dataset_samples.xlsx')
    try:
        save_samples_to_excel(data, info, num_samples, n_show, desc_map, out_file)
    except Exception as e:
        print(f"保存到 Excel 时出错: {e}")


if __name__ == '__main__':
    main()

