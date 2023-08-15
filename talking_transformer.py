# started with a copy of voxel_transformer then updated for finetuning tasks
# currently set up for same-timbre binary-task only
import torch
import torch.nn as nn
from random import randint
import numpy as np
from Constants import ATTENTION_HEADS, env, EPOCHS
print_intermediate=True

class SelfAttention(nn.Module):
    def __init__(self, voxel_dim, heads):
        super(SelfAttention, self).__init__()
        self.voxel_dim = voxel_dim
        self.heads = heads
        self.head_dim = voxel_dim // heads

        assert (self.head_dim * heads == voxel_dim)

        self.values_heads = []
        self.keys_heads = []
        self.queries_heads = []

        #Multi head attention maps
        for i in range(0, self.heads):

            self.values_heads.append(nn.Linear(self.head_dim, self.head_dim, bias=True))
            self.keys_heads.append(nn.Linear(self.head_dim, self.head_dim, bias=True))
            self.queries_heads.append(nn.Linear(self.head_dim, self.head_dim, bias=True))

        self.fc_out = nn.Linear(self.heads * self.head_dim, voxel_dim)

    def forward(self, values, keys, query, mask, sa_print_flag=0):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        #Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        #need to clone our tensors to preserve gradient for some reason, pretty confusing error without these
        new_values = values.clone()
        new_keys = keys.clone()
        new_queries = queries.clone()

        #Messy looking loop but wouldn't work with python-ese attempts
        if True:
            for batch_idx in range(0,N):
                for element in range(0,value_len):
                    for i in range(0, self.heads):
                        new_values[batch_idx][element][i] = self.values_heads[i](values[batch_idx][element][i])

                for element in range(0, key_len):
                    for i in range(0, self.heads):
                        new_keys[batch_idx][element][i] = self.keys_heads[i](keys[batch_idx][element][i].clone())

                for element in range(0, query_len):
                    for i in range(0, self.heads):
                        new_queries[batch_idx][element][i] = self.queries_heads[i](queries[batch_idx][element][i])

        energy = torch.einsum("nqhd,nkhd->nhqk", [new_queries,new_keys])

        if(sa_print_flag):
            #print("new queries is "+str(new_queries)+" and has shape "+str(new_queries.shape))
            #print("new keys is "+str(new_keys)+" and has shape "+str(new_keys.shape))
            print("new values is "+str(new_values)+" and has shape "+str(new_values.shape))
            print("energy is "+str(energy))
            print("in SA mask is "+str(mask))
        # queries shape: (N, query_len, heads, heads_dim)
        # keys shape: (N, key_len, heads, heads_dim)
        # energy shape: (N, heads, query_len, key_len)

        if mask is not None:
            #print("energy has shape "+str(energy.shape))
            #print("mask has shape "+str(mask.shape))
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        if(sa_print_flag):
            print("after filling, energy is "+str(energy))
        temp = energy / (self.voxel_dim ** (1 / 2))
        if(sa_print_flag):
            print("temp is "+str(temp))
        attention = torch.softmax(temp, dim=3)

        print(attention)


        out = torch.einsum("nhql,nlhd->nqhd",[attention, new_values]).reshape(
            N, query_len, self.heads*self.head_dim
        )


        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out shape: (N, query_len, heads, head_dim) then flatten last two dimensions
        if(sa_print_flag):
            print("inside SA, attention is "+str(attention))
            print("before SA fc layer, out is "+str(out))
        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, voxel_dim, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        #self.attention = SelfAttention(voxel_dim, heads)
        self.attention = nn.MultiheadAttention(voxel_dim, heads, batch_first=True)

        self.norm1 = nn.LayerNorm(voxel_dim)
        self.norm2 = nn.LayerNorm(voxel_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(voxel_dim, forward_expansion * voxel_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion * voxel_dim, voxel_dim)
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self, value, key, query, mask, block_print_flag=0):
        attention, attn_weights = self.attention(value, key, query, average_attn_weights=False)


        if(block_print_flag):
            print("value is "+str(value))
            print("key is "+str(key))
            print("query is "+str(query))
            print("mask is "+str(mask))
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        if(block_print_flag):
            print("in block's forward:\n")
            print("value is "+str(value))
            print("attention is "+str(attention))
            print("attention has shape "+str(attention.shape))
            print("x  is "+str(x))
            print("forward is "+str(forward))
            print("out is "+str(out))
        return out, attn_weights
        #return out, attention

class Encoder(nn.Module):
    def __init__(self,
                 voxel_dim,
                 num_layers,
                 heads,
                 device,
                 forward_expansion,
                 dropout,
                 max_length,
                    ):
        super(Encoder,self).__init__()
        self.voxel_dim = voxel_dim
        self.device = device
        self.position_embedding = nn.Embedding(max_length, voxel_dim)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    voxel_dim,
                    heads,
                    dropout=dropout,
                    forward_expansion = forward_expansion,
                )
            for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, encoder_print_flag):
        N, seq_length, voxel_dim = x.shape
        #print("seq_length is "+str(seq_length))
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        #print("positions is "+str(positions))

        embedded_positions = self.position_embedding(positions)

        #print("embedded_positions is "+str(embedded_positions))
        #print("x has shape "+str(x.shape))
        out = self.dropout(x + embedded_positions)
        if(encoder_print_flag):
            print("In Encoder's forward, x is "+str(x))
            print("positions is "+str(positions))
            print("embedded positions is "+str(embedded_positions))

        attn_weights_per_layer = []
        for layer in self.layers:
            out, attn_weights = layer(out, out, out, mask, encoder_print_flag)
            attn_weights_per_layer.append(attn_weights)
            if(encoder_print_flag):
                print("out for layer "+str(layer)+" is "+str(out))
        first_attn_weights = attn_weights_per_layer[0]
        second_attn_weights = attn_weights_per_layer[1]
        third_attn_weights = attn_weights_per_layer[2]
        return out, first_attn_weights, second_attn_weights, third_attn_weights

class DecoderBlock(nn.Module):
    def __init__(self, voxel_dim, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(voxel_dim, heads)
        self.norm = nn.LayerNorm(voxel_dim)
        self.transformer_block = TransformerBlock(
            voxel_dim, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        y = torch.clone(x)
        z = torch.clone(x)
        dec_attention = self.attention(x, y, z, trg_mask)
        dec_query = self.dropout(self.norm(dec_attention + x))
        dec_out = self.transformer_block(value, key, dec_query, src_mask)
        return dec_out

class Decoder(nn.Module):
    def __init__(
            self,
            next_sequence_labels,
            num_genres,
            voxel_dim,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.position_embedding = nn.Embedding(max_length, voxel_dim)

        self.layers = nn.ModuleList(
            [DecoderBlock(voxel_dim, heads, forward_expansion, dropout, device)
             for _ in range(num_layers)]

        )

        self.fc_out = nn.Linear(voxel_dim, next_sequence_labels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        #print("In the decoder's forward call, x is "+str(x))
        N, seq_length, voxel_dim = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        embedded_positions = self.position_embedding(positions)
        x = self.dropout(x + embedded_positions)

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)

        return out

class Transformer(nn.Module):
    def __init__(
            self,
            num_CLS_labels,
            num_genres,
            src_pad_sequence,
            voxel_dim=8,
            num_layers=2,
            forward_expansion=4,
            heads=ATTENTION_HEADS,
            dropout=0.1,
            #device="cuda",
            device="cpu",
            max_length=None,
            ref_samples=None,
            mask_task=None,
            print_flag=0
    ):
        self.mask_task=mask_task
        self.print_flag=print_flag
        super(Transformer, self).__init__()
        print("Model has " + str(heads) + " many attention heads and " + str(
            num_layers) + " many layers and a forward expansion factor of " + str(forward_expansion))

        self.encoder = Encoder(
            voxel_dim,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.decoder = Decoder(
            num_CLS_labels,
            num_genres,
            voxel_dim,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
        )

        self.output_layer_bin = nn.Sequential(
            nn.Linear(voxel_dim, voxel_dim//2),
            nn.Linear(voxel_dim//2, num_CLS_labels),
            nn.Softmax(dim=1)
            #nn.ReLU()

        )
        self.output_layer_genredecoding =  nn.Sequential(
            nn.Linear(voxel_dim, voxel_dim // 2),
            #nn.ReLU(),
            nn.Linear(voxel_dim//2, num_genres),
            nn.Softmax(dim=1)
        )
        self.output_layer_reconstruction =  nn.Sequential(
            nn.Linear(voxel_dim, voxel_dim),
            nn.ReLU(),
            nn.Linear(voxel_dim, voxel_dim),
        )
        self.output_layer_finetune = nn.Sequential(
            nn.Linear(voxel_dim,num_CLS_labels),
            nn.Softmax(dim=1)
            #nn.ReLU()

        )
        self.output_layer_bin_switch = nn.Sequential(
            nn.Linear(voxel_dim,num_CLS_labels),
            nn.Softmax(dim=1)
            #nn.ReLU()

        )
        self.output_layer_finetune_switch = nn.Sequential(
            nn.Linear(voxel_dim, voxel_dim//2),
            nn.Linear(voxel_dim//2, num_CLS_labels),
            nn.Softmax(dim=1)
            #nn.ReLU()

        )

        self.src_pad_sequence = src_pad_sequence
        self.device = device

    def make_src_mask(self, src):
        #print("src has shape "+str(src.shape))
        src_mask = []
        N = src.shape[0]
        seq_len = src.shape[1]
        voxel_dim = src.shape[2]
        src_pad_sequence = [0] * voxel_dim

        #batch is the index within the batch, not the actual batch number wrt the training set
        for batch in range(0, N):
            src_mask.append([])
            sequence = src[batch] #this is a sequence of vectors in voxel space
            for token in range(0, seq_len):
                #print("this element of the sequence is "+str(sequence[token])+" and srcpadseq is "+str(src_pad_sequence))
                maskbool = (sequence[token].tolist()==src_pad_sequence)
                #print("therefore maskbool is "+str(maskbool))
                src_mask[batch].append(maskbool)

        src_mask = torch.tensor(src_mask).unsqueeze(1).unsqueeze(2)
        #print("src mask is "+str(src_mask))
        #print("src mask shape is "+str(src_mask.shape))
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    # def make_trg_mask(self, trg):
    #     N, trg_len, voxel_dim = trg.shape
    #     trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
    #         N, 1, trg_len, trg_len
    #     )
    #     return trg_mask.to(self.device)

    def forward(self, src, mask_indices):
        if(self.print_flag):
            print("src is "+str(src)+" and mask indices is "+str(mask_indices))
        src_mask = self.make_src_mask(src)
        #trg_mask = None
        enc_src, first_attn_weights, second_attn_weights, third_attn_weights = self.encoder(src, src_mask, self.print_flag)
        #out = self.decoder(trg, enc_src, src_mask, trg_mask)
        #print("enc_src has shape "+str(enc_src.shape))
        batch_CLS_tokens = enc_src[:,0,:] #slice out the first element of each transformed sequence (the final form of CLS token)
        if(mask_indices not in ["finetune", "sametimbre", "finetune_switch", "sametimbre_switch"]):
            batch_MSK_tokens = []

            BATCHSIZE=len(mask_indices)
            voxel_dim=0 #just to get rid of a warning, gets overwritten in for loop below
            #make a list of the final states of the MSK tokens
            #the mask_indices list tells us where to find them
            for i in range(0, BATCHSIZE):
                sample_mask_idxs=mask_indices[i] #a list of either two or one idxs, depending whether mask variation was true
                sample_mask_tokens=[]
                for mask_idx in sample_mask_idxs:
                    if(mask_idx==-1):
                        continue
                    temp = enc_src[i][mask_idx][:]
                #print("in Transformer's forward, temp is "+str(temp)+" and batchmsktokens is "+str(batch_MSK_tokens))
                    batch_MSK_tokens.append(temp)

            batch_MSK_tokens = torch.stack(batch_MSK_tokens) #create pytorch tensor of the tensors in the list
        if(self.print_flag):
            print("output of encoder stacks is "+str(enc_src))
            print("batch CLS tokens are "+str(batch_CLS_tokens))
            print("batch MSK tokens are "+str(batch_MSK_tokens))
        #print("batch MSK tokens has shape "+str(batch_MSK_tokens.shape))  #should be batchsize by voxel_dim

        if(mask_indices=="finetune"):
            out_finetune=self.output_layer_finetune(batch_CLS_tokens)
            return out_finetune, None, first_attn_weights, second_attn_weights, third_attn_weights
        elif(mask_indices=="sametimbre"):
            out_bin=self.output_layer_bin(batch_CLS_tokens)
            return out_bin
        elif(mask_indices=="finetune_switch"): # to try a different composition of the output layer
            out_finetune=self.output_layer_finetune_switch(batch_CLS_tokens)
            return out_finetune
        elif(mask_indices=="sametimbre_switch"):
            out_bin=self.output_layer_bin_switch(batch_CLS_tokens)
            return out_bin
        else:
            #out_bin=self.output_layer_bin(batch_CLS_tokens)
            out_bin=self.output_layer_bin_switch(batch_CLS_tokens)

            if(self.mask_task=="genre_decoding"):
                out_multi=self.output_layer_genredecoding(batch_MSK_tokens)
            elif(self.mask_task=="reconstruction"):
                out_multi=self.output_layer_reconstruction(batch_MSK_tokens)
            return out_bin, out_multi, first_attn_weights, second_attn_weights, third_attn_weights

def get_mask_idx(sequence, src_pad_sequence): #n-by-voxel_dim sequence of voxel data
    padded_len = len(sequence)
    pad_idx = padded_len

    #mask token is placed randomly, but we don't want to mask padding
    for i in range(0,padded_len):
        if sequence[i] == src_pad_sequence: #if we've reached the left-boundary of the padding
            pad_idx = i #pad_idx is now the index of padding's left-boundary
            break

    mask_idx = randint(1, pad_idx-1) #randint is inclusive of both parameters, dont want to mask cls_token at index 0, don't want to mask padding

    return mask_idx


# if __name__ == "__main__":
#     ##############################  SET PARAMETERS  ##############################
#     #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     device = "cpu"
#     MSK_flag = 0
#     CLS_flag = 1
#
#     next_sequence_labels = 2 #two possible labels for next sequence prediction task, yes or no
#     max_length = 8 #length of voxel sequence*2 + 1 (CLS token) + 1 (SEP token)
#     actual_voxel_dim = 5
#     voxel_dim = actual_voxel_dim+3 #add dimensions to flag CLS, MSK, and SEP
#     num_genres = 8 #from the training set
#
#     src_pad_sequence = [0]*voxel_dim
#     CLS_token = [1]+([0]*(voxel_dim-1)) #first dimension is reserved for cls_token flag
#     MSK_token = [0, 1]+([0]*(voxel_dim-2)) #second dimension is reserved for msk_token flag
#     SEP_token = [0, 0, 1]+([0]*(voxel_dim-3)) #third dimension is reserved for sep_token flag
#     cross_entropy_loss = nn.CrossEntropyLoss()
#     kl_loss = nn.KLDivLoss(reduction='batchmean')
#
#     t00 = [0, 0, 0, 4, 5, 3, 2, 1]  # list of voxels in sample 0 at time 0
#     t01 = [0, 0, 0, 4, 5, 3, 2, 1]  # list of voxels in sample 0 at time 2
#     t02 = [0, 0, 0, 0, 0, 0, 0, 0]  # list of voxels in sample 0 at time 1
#     t10 = [0, 0, 0, 9, 2, 3, 4, 0]  # list of voxels in sample 1 at time 0
#     t11 = [0, 0, 0, 7, 1, 2, 3, 6]  # list of voxels in sample 1 at time 1
#     t12 = [0, 0, 0, 4, 5, 3, 2, 1]  # list of voxels in sample 1 at time 2
#     fake =[0, 0, 0, -1, -1, -1, -1, -1]
#
#
#     sample0 = [CLS_token, t00, t01, t02, SEP_token, t02, t01, t00]
#     sample1 = [CLS_token, t10, t11, t12, SEP_token, fake, fake, fake]
#
#     #labels for first batch
#     batch0 = torch.tensor([sample0, sample1])
#     genre_labels0 = torch.tensor([4, 5])
#     next_sequence_labels0 = torch.tensor([1, 0])
#     #targets0 = torch.tensor([labels0, labels1])
#
#     batch_size = len(batch0)
#
#     if(MSK_flag):
#         for sample in range(0,batch_size):
#             mask_idx = get_mask_idx(batch0[sample], src_pad_sequence)
#             batch0[sample][mask_idx] = torch.tensor(MSK_token)
#             # print("For sample "+str(sample)+", masked index "+str(mask_idx))
#             # print("That sequence is now  "+str(batch0[sample]))
#
#     model = Transformer(next_sequence_labels=next_sequence_labels, num_genres=num_genres, src_pad_sequence=src_pad_sequence, max_length=max_length, voxel_dim=voxel_dim).to(device)
#     out = model(batch0, None) #inputs to Transformer class's forward pass are inputs to encoder, and inputs to decoder
#     print(out.shape)
#     print(str(out))
#
#     loss = cross_entropy_loss(out, next_sequence_labels0)
#     loss.backward()
#     print("Loss: "+str(loss))
