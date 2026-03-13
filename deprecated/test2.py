from voxel_transformer import *
from pitchclass_data import *
from torch.utils.data import DataLoader
import torch.optim as optim

if __name__ == "__main__":
    ##############################  SET PARAMETERS  ##############################
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    MSK_flag = 0
    CLS_flag = 1
    BATCH_SIZE = 2
    EPOCHS = 5
    LEARNING_RATE = 0.001

    unpadded_samples = load_subject("1088")

    padded_samples = make_padded_samples(unpadded_samples)

    seq_len = 6
    TIMESTEPS = len(padded_samples)
    voxel_dim = len(padded_samples[0])
    CLS_token = [1] + ([0] * (voxel_dim - 1))  # first dimension is reserved for cls_token flag
    MSK_token = [0, 1] + ([0] * (voxel_dim - 2))  # second dimension is reserved for msk_token flag
    SEP_token = [0, 0, 1] + ([0] * (voxel_dim - 3))  # third dimension is reserved for sep_token flag
    NS_X, NS_y = make_NS_data(padded_samples, seq_len, TIMESTEPS, voxel_dim, CLS_token, SEP_token) #get tensors for samples and labels
    train_data = TrainData(NS_X, NS_y) #make the TrainData object
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True) #make the DataLoader object
    next_sequence_labels = 2 #two possible labels for next sequence prediction task, yes or no
    max_length = 14 #length of voxel sequence*2 + 1 (CLS token) + 1 (SEP token)


    num_genres = 8 #from the training set

    src_pad_sequence = [0]*voxel_dim

    cross_entropy_loss = nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction='batchmean')


    if(MSK_flag):
        for sample in range(0,batch_size):
            mask_idx = get_mask_idx(batch0[sample], src_pad_sequence)
            batch0[sample][mask_idx] = torch.tensor(MSK_token)
            # print("For sample "+str(sample)+", masked index "+str(mask_idx))
            # print("That sequence is now  "+str(batch0[sample]))

    model = Transformer(next_sequence_labels=next_sequence_labels, num_genres=num_genres, src_pad_sequence=src_pad_sequence, max_length=max_length, voxel_dim=voxel_dim).to(device)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train() #sets model status, doesn't actually start training
    for e in range(1, EPOCHS+1):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            y_pred = model(X_batch)
            #print("y_pred has shape "+str(y_pred.shape))
            #print("y_batch has shape "+str(y_batch.shape))
            loss = criterion(y_pred, y_batch)
            #acc = binary_acc(y_pred, y_batch)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            #epoch_acc += acc.item()

        #print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')
        print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f}')

    #model = torch.load("/Users/sean/Desktop/current_research/fmriBERTfix/fmriBERT/pitchclass/experiment1.pt")
    #model.eval()

    out = model(NS_X[4:6])
    print(out)
