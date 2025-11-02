import os
import time
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import netAll
import lossFunction

class SimpleDataset(Dataset):
    def __init__(self, UUMat, DUMat, INMat, TAMat, CIMat, noise2DU, noise2BS, alpha_SI):
        self.UUMat = UUMat
        self.DUMat = DUMat
        self.INMat = INMat
        self.TAMat = TAMat
        self.CIMat = CIMat
        self.noise2DU = noise2DU
        self.noise2BS = noise2BS
        self.alpha_SI = alpha_SI

    def __len__(self):
        return self.UUMat.shape[0]

    def __getitem__(self, idx):
        return (self.UUMat[idx], self.DUMat[idx], self.INMat[idx], self.TAMat[idx],
                self.CIMat[idx], self.noise2DU[idx], self.noise2BS[idx], self.alpha_SI)


def try_load_dataset(folder):
    ds_path = os.path.join(folder, 'dataset.pt')
    info_path = os.path.join(folder, 'data_info.pt')
    if os.path.exists(ds_path):
        data = torch.load(ds_path, map_location='cpu',weights_only=False)
        return data
    raise FileNotFoundError(f"No dataset file at {ds_path}")

def collate_fn(batch):
    # batch is list of tuples, stack them
    cols = list(zip(*batch))
    return tuple(torch.stack(c, dim=0) if isinstance(c[0], torch.Tensor) else torch.stack([torch.tensor(c_i) for c_i in c], dim=0) for c in cols)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    lossFunction.device = device

    workspace_folder = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(workspace_folder, 'dataset')
    data = try_load_dataset(data_folder)
    def pick(d, keys):
        for k in keys:
            if k in d:
                return d[k]
        return None
    UUMat = pick(data, ['UUMat'])
    DUMat = pick(data, ['DUMat'])
    INMat = pick(data, ['INMat'])
    TAMat = pick(data, ['TAMat'])
    CIMat = pick(data, ['CIMat'])
    noise2DU = pick(data, ['noise2DU'])
    noise2BS = pick(data, ['noise2BS'])
    alpha_SI = pick(data, ['alpha_SI'])

    if UUMat is None or DUMat is None or INMat is None or TAMat is None or CIMat is None:
        raise KeyError('Dataset file missing expected tensors; please provide a dataset with UUMat/DUMat/INMat/TAMat/CIMat')

    UUMat = UUMat.to(device)
    DUMat = DUMat.to(device)
    INMat = INMat.to(device)
    TAMat = TAMat.to(device)
    CIMat = CIMat.to(device)

    batch_size = UUMat.shape[0]
    num_trans = DUMat.shape[2]
    num_rece = UUMat.shape[2]

    # Convert scalar noise/alpha values to indexable tensors if necessary
    # noise2DU -> shape (batch, L, 1)
    if isinstance(noise2DU, torch.Tensor):
        noise2DU = noise2DU.to(device)
        if noise2DU.numel() == 1:
            noise2DU = noise2DU.view(1, 1, 1).expand(batch_size, DUMat.shape[1], 1).contiguous()
        elif noise2DU.dim() == 2 and noise2DU.shape[0] == DUMat.shape[1]:
            noise2DU = noise2DU.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        elif noise2DU.dim() == 1 and noise2DU.shape[0] == DUMat.shape[1]:
            noise2DU = noise2DU.view(1, DUMat.shape[1], 1).expand(batch_size, -1, -1).contiguous()
    elif isinstance(noise2DU, (float, int)):
        noise2DU = torch.full((batch_size, DUMat.shape[1], 1), float(noise2DU), dtype=torch.float32, device=device)
    else:
        print("error: noise2DU is neither tensor nor scalar")

    # noise2BS -> shape (batch,1,1)
    if isinstance(noise2BS, torch.Tensor):
        noise2BS = noise2BS.to(device)
        if noise2BS.numel() == 1:
            noise2BS = noise2BS.view(1, 1, 1).expand(batch_size, -1, -1).contiguous()
    elif isinstance(noise2BS, (float, int)):
        noise2BS = torch.full((batch_size, 1, 1), float(noise2BS), dtype=torch.float32, device=device)
    else:
        print("error: noise2BS is neither tensor nor scalar")

    # alpha_SI -> scalar tensor
    if isinstance(alpha_SI, torch.Tensor):
        alpha_SI = alpha_SI.to(device)
    elif isinstance(alpha_SI, (float, int)):
        alpha_SI = torch.tensor(float(alpha_SI), dtype=torch.float32, device=device)
    else:
        print("error: alpha_SI is neither tensor nor scalar")

    dataset = SimpleDataset(UUMat, DUMat, INMat, TAMat, CIMat, noise2DU, noise2BS, alpha_SI)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # create model and loss
    model = netAll.netAll(num_trans, num_rece, args.num_heads, args.embed_dim).to(device)
    loss_fn = lossFunction.LossFunction(lambda1=args.lambda1)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print('Starting training: epochs=', args.epochs, 'batch_size=', args.batch_size)
    print("threads:", torch.get_num_threads())
    start_time = time.time()
    model.train()
    for epoch in range(args.epochs):
        print('the echo is', epoch)
        epoch_loss = 0.0
        iters = 0
        for batch in loader:
            # batch is tuple stacked by collate
            UUMat_b, DUMat_b, INMat_b, TAMat_b, CIMat_b, noise2DU_b, noise2BS_b, alpha_SI_b = batch
            UUMat_b = UUMat_b.to(device)
            DUMat_b = DUMat_b.to(device)
            INMat_b = INMat_b.to(device)
            TAMat_b = TAMat_b.to(device)
            CIMat_b = CIMat_b.to(device)
            noise2DU_b = noise2DU_b.to(device)
            noise2BS_b = noise2BS_b.to(device)
            # alpha_SI_b may be tensor or scalar per-sample; take first if needed
            if isinstance(alpha_SI_b, torch.Tensor):
                alpha_SI_val = alpha_SI_b[0] if alpha_SI_b.numel() > 1 else alpha_SI_b
            else:
                alpha_SI_val = alpha_SI_b

            optimizer.zero_grad()
            # forward
            try:
                UUPowerMat, DUComMat, SensingMat = model(UUMat_b, DUMat_b, INMat_b, TAMat_b, CIMat_b)
            except Exception as e:
                print('Model forward failed:', e)
                raise

            loss = loss_fn(UUPowerMat, DUComMat, SensingMat,
                           UUMat_b, DUMat_b, TAMat_b, INMat_b,
                           CIMat_b, noise2DU_b, noise2BS_b, alpha_SI_val,
                           num_trans, num_rece)

            # backprop
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            iters += 1
            if args.smoke:
                break

        avg_loss = epoch_loss / max(1, iters)
        print(f'Epoch {epoch+1}/{args.epochs}  avg_loss={avg_loss:.6f}')
        if args.smoke:
            break

    elapsed = time.time() - start_time
    print(f'Training finished in {elapsed:.1f}s')

    # save checkpoint
    ckpt = {'model_state_dict': model.state_dict(), 'args': vars(args)}
    save_path = args.save_path or os.path.join(workspace_folder, 'netAll_checkpoint.pt')
    torch.save(ckpt, save_path)
    print('Saved checkpoint to', save_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--embed-dim', type=int, default=4)
    parser.add_argument('--num-trans', type=int, default=8)
    parser.add_argument('--num-rece', type=int, default=6)
    parser.add_argument('--num-heads', type=int, default=2)
    parser.add_argument('--lambda1', type=float, default=0.1)
    parser.add_argument('--save-path', type=str, default='')
    parser.add_argument('--smoke', action='store_true', help='run only a single batch and exit')
    # dataset fallback sizes
    parser.add_argument('--num-uu', type=int, default=4)
    parser.add_argument('--num-du', type=int, default=3)
    parser.add_argument('--num-in', type=int, default=5)
    parser.add_argument('--num-ta', type=int, default=2)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
