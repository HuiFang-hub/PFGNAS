from src.FedPUB.misc.utils import *

class DataLoader:
    def __init__(self, args):
        self.args = args
        self.n_workers = 1
        self.client_id = None

        from torch_geometric.loader import DataLoader
        self.DataLoader = DataLoader

    def switch(self, client_id):
        if not self.client_id == client_id:
            self.client_id = client_id
            self.partition = get_data(self.args, client_id=client_id)
            self.pa_loader = self.DataLoader(dataset=self.partition, batch_size=1, 
                shuffle=False, num_workers=self.n_workers, pin_memory=False)

def get_data(args, client_id):
    # torch_save(data_path, f'{mode}/{n_clients}/partition_{client_id}.pt', {
    #     'client_data': client_data,
    #     'client_id': client_id
    # })

    dataset = args.data.type.lower()
    data_path = os.path.join(args.data.root, f'{dataset}')

    if args.data.splitter == 'lda':
        alpha =args.data.splitter_args[0]['alpha']
        data = [
            torch_load(
                data_path,
                f'{args.data.mode}/{args.federate.client_num}_{alpha}/partition_{client_id}.pt'
            )['client_data']
        ]
    else:
        data = [
            torch_load(
                data_path,
                f'{args.data.mode}/{args.federate.client_num}/partition_{client_id}.pt'
            )['client_data']
        ]


    return data
