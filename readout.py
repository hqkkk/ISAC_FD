import os
import torch
import numpy as np
import pandas as pd
import data_create as dc


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
    print("=== readout.py: 根据 data_create 参数生成 10 个样本并保存到 Excel ===")
    base_dir = os.path.dirname(__file__)

    # 按 data_create 中的参数设置创建 systemParameter
    # 这些默认值可以根据需要调整
    noise2DU_dBm = -70
    noise2BS_dBm = -70
    alpha_SI_dB = -110
    num_trans = 4
    num_rece = 4
    # 与 data_create.py 保持一致：基站功率使用 dBm2watt(18)
    power_BS = dc.dBm2watt(18)
    BSlocation_XYZ = np.array([0, 20, 0])

    ta_num = 2
    ta_angleRange = np.array([[45,75],[105,135]])
    # 与 data_create.py 保持一致：ta_powerGainArr 使用 sqrt(dBm2watt(noise2BS_dBm - 30))
    ta_powerGainArr = np.full((ta_num,), np.sqrt(dc.dBm2watt(noise2BS_dBm - 30)))
    ta_distanceRange = np.array([75,125])#此时数量级较为统一

    in_num = 2
    in_angleRange = np.array([45, 135])
    # 与 data_create.py 保持一致：in_powerGainArr 使用 sqrt(dBm2watt(noise2BS_dBm + 20))
    in_powerGainArr = np.full((in_num,), np.sqrt(dc.dBm2watt(noise2BS_dBm + 20)))
    in_distanceRange = np.array([75,125])#此时数量级较为统一

    uu_num = 4
    # 与 data_create.py 保持一致：uu_powerBudget = dBm2watt(5)
    uu_powerBudget = dc.dBm2watt(5)
    uu_locatRange = np.array([[0, 100], [0, 10], [0, 100]])

    du_num = 4
    du_locatRange = np.array([[0, 100], [0, 10], [0, 100]])

    BS = dc.BSparameter(num_trans, num_rece, power_BS, BSlocation_XYZ)
    TA = dc.TAparameter(ta_num, ta_angleRange, ta_powerGainArr, ta_distanceRange)
    IN = dc.INparameter(in_num, in_angleRange, in_powerGainArr, in_distanceRange)
    UU = dc.UUparameter(uu_num, uu_powerBudget, uu_locatRange)
    DU = dc.DUparameter(du_num, du_locatRange)
    system = dc.systemParameter(BS, TA, IN, UU, DU)
    # 与 data_create.py 保持一致：noise2* 在某处被使用为 sqrt(dBm2watt(...))，因此这里传入相同的值
    system.envParaSet(np.sqrt(dc.dBm2watt(noise2DU_dBm)), np.sqrt(dc.dBm2watt(noise2BS_dBm)), dc.dB2linear(alpha_SI_dB), num_trans, num_rece)

    n_samples = 10
    samples = []
    for i in range(n_samples):
        # system.data_create 内会随机生成场景
        UUMat, DUMat, INMat, TAMat, CIMat = system.data_create()
        # 拷贝参数字典，避免后续随机更新覆盖之前的样本记录
        def copy_params(d):
            out = {}
            for k, v in d.items():
                try:
                    if isinstance(v, np.ndarray):
                        out[k] = v.copy()
                    else:
                        out[k] = v
                except Exception:
                    out[k] = v
            return out

        samples.append({
            'UUMat': UUMat.detach().cpu().numpy(),
            'DUMat': DUMat.detach().cpu().numpy(),
            'INMat': INMat.detach().cpu().numpy(),
            'TAMat': TAMat.detach().cpu().numpy(),
            'CIMat': CIMat.detach().cpu().numpy(),
            'system': {
                'num_trans': num_trans,
                'num_rece': num_rece,
                'powerBudget': power_BS,
                'location': BSlocation_XYZ.reshape(-1, 1)
            },
            'TA_params': copy_params(TA.__dict__),
            'IN_params': copy_params(IN.__dict__),
            'UU_params': copy_params(UU.__dict__),
            'DU_params': copy_params(DU.__dict__),
            'env': {
                # 与 data_create.py 对齐：噪声使用 sqrt(dBm2watt(...))
                'noise2DU_W': np.sqrt(dc.dBm2watt(noise2DU_dBm)),
                'noise2BS_W': np.sqrt(dc.dBm2watt(noise2BS_dBm)),
                'alpha_SI_linear': dc.dB2linear(alpha_SI_dB)
            }
        })

    # 写入 Excel
    out_file = os.path.join(base_dir, 'generated_10_samples.xlsx')
    with pd.ExcelWriter(out_file, engine='xlsxwriter') as writer:
        # summary sheet
        # 与 data_create.py 对齐：噪声项使用 sqrt(dBm2watt(...))，alpha 使用线性值
        summary = {
            'num_samples': n_samples,
            'num_trans': num_trans,
            'num_rece': num_rece,
            'power_BS_W': power_BS,
            'noise2DU_W': np.sqrt(dc.dBm2watt(noise2DU_dBm)),
            'noise2BS_W': np.sqrt(dc.dBm2watt(noise2BS_dBm)),
            'alpha_SI_linear': dc.dB2linear(alpha_SI_dB)
        }
        pd.DataFrame(list(summary.items()), columns=['key', 'value']).to_excel(writer, sheet_name='Summary', index=False)

        for idx, s in enumerate(samples):
            sheet = f"Sample_{idx+1}"
            row = 0
            # system params
            pd.DataFrame(["系统参数"]).to_excel(writer, sheet_name=sheet, startrow=row, index=False, header=False)
            row += 1
            pd.DataFrame(list(s['system'].items()), columns=['key', 'value']).to_excel(writer, sheet_name=sheet, startrow=row, index=False)
            row += len(s['system']) + 2

            # TA/IN/UU/DU params
            pd.DataFrame(["TA参数"]).to_excel(writer, sheet_name=sheet, startrow=row, index=False, header=False)
            row += 1
            try:
                pd.DataFrame(list(s['TA_params'].items()), columns=['key', 'value']).to_excel(writer, sheet_name=sheet, startrow=row, index=False)
                row += len(s['TA_params']) + 1
            except Exception:
                pass

            pd.DataFrame(["IN参数"]).to_excel(writer, sheet_name=sheet, startrow=row, index=False, header=False)
            row += 1
            try:
                pd.DataFrame(list(s['IN_params'].items()), columns=['key', 'value']).to_excel(writer, sheet_name=sheet, startrow=row, index=False)
                row += len(s['IN_params']) + 1
            except Exception:
                pass

            pd.DataFrame(["UU参数"]).to_excel(writer, sheet_name=sheet, startrow=row, index=False, header=False)
            row += 1
            try:
                pd.DataFrame(list(s['UU_params'].items()), columns=['key', 'value']).to_excel(writer, sheet_name=sheet, startrow=row, index=False)
                row += len(s['UU_params']) + 1
            except Exception:
                pass

            pd.DataFrame(["DU参数"]).to_excel(writer, sheet_name=sheet, startrow=row, index=False, header=False)
            row += 1
            try:
                pd.DataFrame(list(s['DU_params'].items()), columns=['key', 'value']).to_excel(writer, sheet_name=sheet, startrow=row, index=False)
                row += len(s['DU_params']) + 2
            except Exception:
                pass

            # env
            pd.DataFrame(["环境参数"]).to_excel(writer, sheet_name=sheet, startrow=row, index=False, header=False)
            row += 1
            pd.DataFrame(list(s['env'].items()), columns=['key', 'value']).to_excel(writer, sheet_name=sheet, startrow=row, index=False)
            row += len(s['env']) + 2

            # Channel matrices: write real and imag separately
            def write_complex_matrix(name, mat):
                nonlocal row
                mat = np.array(mat)
                real_df = pd.DataFrame(mat.real)
                imag_df = pd.DataFrame(mat.imag)
                pd.DataFrame([f"{name} (real)"]).to_excel(writer, sheet_name=sheet, startrow=row, index=False, header=False)
                row += 1
                real_df.to_excel(writer, sheet_name=sheet, startrow=row, index=False)
                row += len(real_df) + 1
                pd.DataFrame([f"{name} (imag)"]).to_excel(writer, sheet_name=sheet, startrow=row, index=False, header=False)
                row += 1
                imag_df.to_excel(writer, sheet_name=sheet, startrow=row, index=False)
                row += len(imag_df) + 2

            write_complex_matrix('UUMat', s['UUMat'])
            write_complex_matrix('DUMat', s['DUMat'])
            write_complex_matrix('INMat', s['INMat'])
            write_complex_matrix('TAMat', s['TAMat'])
            write_complex_matrix('CIMat', s['CIMat'])

    print(f"已生成并保存 {n_samples} 个样本到 {out_file}")


if __name__ == '__main__':
    main()

