import torch.optim as optim
from torch.utils import data

from e2efold.models import ContactNetwork, ContactNetwork_test, ContactNetwork_fc
from e2efold.models import ContactAttention, ContactAttention_simple_fix_PE
from e2efold.models import Lag_PP_NN, RNA_SS_e2e, Lag_PP_zero, Lag_PP_perturb, Lag_PP_final
from e2efold.models import Lag_PP_mixed, ContactAttention_simple
from e2efold.common.utils import *
from e2efold.common.config import process_config
import e2efold.evaluation
from e2efold.common.long_seq_pre_post_process import combine_chunk_u_maps_no_replace
from e2efold.postprocess import postprocess

# START Config Setup:
args = get_args()
config_file = args.config
config = process_config(config_file)
print("#####Stage 3#####")
print('Here is the configuration of this run: ')
print(pretty_obj(config))

d                        = config.u_net_d
nameof_exper             = config.exp_name
BATCH_SIZE               = config.BATCH_SIZE
OUT_STEP                 = config.OUT_STEP
LOAD_MODEL               = config.LOAD_MODEL    
pp_steps                 = config.pp_steps
pp_loss                  = config.pp_loss
data_type                = config.data_type
model_type               = config.model_type
pp_type                  = '{}_s{}'.format(config.pp_model, pp_steps)
rho_per_position         = config.rho_per_position
model_path               = config.data_root+'models_ckpt/supervised_{}_{}_d{}_l3.pt'.format(model_type, data_type,d)
pp_model_path            = config.data_root+'models_ckpt/lag_pp_{}_{}_{}_position_{}.pt'.format(pp_type, data_type, pp_loss,rho_per_position)
e2e_model_path           = config.data_root+'models_ckpt/e2e_{}_{}_d{}_{}_{}_position_{}.pt'.format(model_type, pp_type,d, data_type, pp_loss,rho_per_position)
epoches_third            = config.epoches_third
evaluate_epi             = config.evaluate_epi
step_gamma               = config.step_gamma
k                        = config.k
cond_save_ct_predictions = config.save_ct_predictions

os.environ["CUDA_VISIBLE_DEVICES"]= config.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_torch(0)
# END Config Setup

# START Data Loading
from e2efold.data_generator import Dataset_1800, RNASSDataGenerator, Dataset
import collections
RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq ss_label length name pairs')

train_data_600  =                                     RNASSDataGenerator(config.data_root+'data/{}/'.format(data_type), 'train_600')
val_data_600    =                                     RNASSDataGenerator(config.data_root+'data/{}/'.format(data_type), 'val_600')
train_data_1800 =                                     RNASSDataGenerator(config.data_root+'data/{}/'.format(data_type), 'train_1800')
val_data_1800   =                                     RNASSDataGenerator(config.data_root+'data/{}/'.format(data_type), 'val_1800')
if (data_type == 'archiveII_all'):   test_data_1800 = RNASSDataGenerator(config.data_root+'data/{}/'.format(data_type), 'test_1800')
if (data_type == 'rnastralign_all'): test_data_1800 = RNASSDataGenerator(config.data_root+'data/{}/'.format(data_type), 'test_no_redundant_1800')

train_maxof_seq_len = train_data_600.maxof_seq_len # this value is used as a part of the model training algrorithm, currently 600, needed as the L param for instantiating the contact_net below
print(f'train_maxof_seq_len: {train_maxof_seq_len}')

test_maxof_seq_len = test_data_1800.maxof_seq_len
print(f'test_maxof_seq_len: {test_maxof_seq_len}')

params               = {'batch_size': BATCH_SIZE, 'shuffle': True, 'num_workers': 6, 'drop_last': True}
train_set_600        = Dataset(train_data_600)
train_generator_600  = data.DataLoader(train_set_600, **params)

val_set_600          = Dataset(val_data_600)
val_generator_600    = data.DataLoader(val_set_600, **params)

params               = {'batch_size': 1,'shuffle': True,'num_workers': 6,'drop_last': False}
train_set_1800       = Dataset_1800(train_data_1800)
train_generator_1800 = data.DataLoader(train_set_1800, **params)

val_set_1800         = Dataset_1800(val_data_1800)
val_generator_1800   = data.DataLoader(val_set_1800, **params)

params               = {'batch_size': 1,'shuffle': False,'num_workers': 6,'drop_last': False}
test_set_1800        = Dataset_1800(test_data_1800)
test_generator_1800  = data.DataLoader(test_set_1800, **params)

# These conditionals check the type of model for which a data path (on disk) to the model checkpoint data was provided
# so that the model checkpoint data can be imported via contact_net.load_state_dict(...) 
# to the respective in-memory Pytorch model that it was trained on and exported from.
# U-Net (utility/unconstrained score network; alternatively called contact network outputing a contact score map)
if (model_type == 'test_lc'):        contact_net = ContactNetwork_test           (d=d, L=train_maxof_seq_len).to(device)
if (model_type == 'att6'):           contact_net = ContactAttention              (d=d, L=train_maxof_seq_len).to(device)
if (model_type == 'att_simple'):     contact_net = ContactAttention_simple       (d=d, L=train_maxof_seq_len).to(device)    
if (model_type == 'att_simple_fix'): contact_net = ContactAttention_simple_fix_PE(d=d, L=train_maxof_seq_len).to(device)
if (model_type == 'fc'):             contact_net = ContactNetwork_fc             (d=d, L=train_maxof_seq_len).to(device)
if (model_type == 'conv2d_fc'):      contact_net = ContactNetwork                (d=d, L=train_maxof_seq_len).to(device)

# need to write the class for the computational graph of lang pp
# PP-Net (note: Lag -> Lagrangian)
if ('nn'      in pp_type): lag_pp_net = Lag_PP_NN     (pp_steps, k).to(device)
if ('zero'    in pp_type): lag_pp_net = Lag_PP_zero   (pp_steps, k).to(device)
if ('perturb' in pp_type): lag_pp_net = Lag_PP_perturb(pp_steps, k).to(device)
if ('mixed'   in pp_type): lag_pp_net = Lag_PP_mixed  (pp_steps, k, rho_per_position).to(device)
if ('final'   in pp_type): lag_pp_net = Lag_PP_final  (pp_steps, k, rho_per_position).to(device)

rna_ss_e2e = RNA_SS_e2e(contact_net, lag_pp_net)

if LOAD_MODEL and os.path.isfile(model_path):
    print('Loading u net model...')
    contact_net.load_state_dict(torch.load(map_location=device, f=model_path))
if LOAD_MODEL and os.path.isfile(pp_model_path):
    print('Loading pp model...')
    lag_pp_net.load_state_dict(torch.load(map_location=device, f=pp_model_path))
if LOAD_MODEL and os.path.isfile(e2e_model_path):
    print('Loading e2e model...')
    rna_ss_e2e.load_state_dict(torch.load(map_location=device, f=e2e_model_path))

all_optimizer = optim.Adam(rna_ss_e2e.parameters())

# "Since, in the contact map, most entries are 0, we used weighted loss and set the positive sample weight as 300"
pos_weight = torch.Tensor([300]).to(device) 
criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(pos_weight = pos_weight)
criterion_mse = torch.nn.MSELoss(reduction='sum')

# END Data Loading

def model_eval_all_test():
    contact_net.eval()
    lag_pp_net.eval()
    result_no_train = list()
    result_no_train_shift = list()
    result_pp = list()
    result_pp_shift = list()

    f1_no_train = list()
    f1_pp = list()
    seq_lens_list = list()

    batch_n = 0
    for seq_embedding_batch, PE_batch, _, comb_index, seq_embeddings, contacts, seq_lens in test_generator_1800:
        print('Batch number: ', batch_n)
        batch_n += 1

        state_pad = torch.zeros(1,2,2).to(device)
        seq_embedding_batch = seq_embedding_batch[0].to(device)
        PE_batch = PE_batch[0].to(device)
        seq_embedding = torch.Tensor(seq_embeddings.float()).to(device)

        with torch.no_grad():
            pred_contacts = contact_net(PE_batch, seq_embedding_batch, state_pad)
            pred_u_map = combine_chunk_u_maps_no_replace(pred_contacts, comb_index, 6).unsqueeze(0) # U-Net ALONE
            a_pred_list = lag_pp_net(pred_u_map, seq_embedding) # U-Net + PP-Net 

        # "Ground Truth" (actual labels of the test data)
        contacts_batch = torch.Tensor(contacts.float()[:,:1800, :1800])
        
        # Evalute Prediction of: U-Net + Raw Postprocessing (=postprocess(...)), NO training of the PP-Net 
        u_no_train                = postprocess(pred_u_map, seq_embedding, 0.01, 0.1, 50, 1.0, True)
        map_no_train              = (u_no_train > 0.5).float()
        
        result_no_train_tmp       = list(map(lambda i: evaluate_exact   (map_no_train.cpu()[i], contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        result_no_train           += result_no_train_tmp
        
        result_no_train_tmp_shift = list(map(lambda i: evaluate_shifted (map_no_train.cpu()[i], contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        result_no_train_shift     += result_no_train_tmp_shift
        
        f1_no_train_tmp           = list(map(lambda i: F1_low_tri       (map_no_train.cpu()[i], contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        f1_no_train               += f1_no_train_tmp

        # Evalute Prediction of: U-Net + training of the PP-Net 
        final_pred       = (a_pred_list[-1].cpu()>0.5).float()
        
        result_tmp       = list(map(lambda i: evaluate_exact   (final_pred.cpu()[i], contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        result_pp        += result_tmp

        result_tmp_shift = list(map(lambda i: evaluate_shifted (final_pred.cpu()[i], contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        result_pp_shift  += result_tmp_shift

        f1_tmp           = list(map(lambda i: F1_low_tri       (final_pred.cpu()[i], contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        f1_pp            += f1_tmp
        
        seq_lens_list += list(seq_lens)

    nt_exact_p,nt_exact_r,nt_exact_f1 = zip(*result_no_train)
    nt_shift_p,nt_shift_r,nt_shift_f1 = zip(*result_no_train_shift)  

    pp_exact_p,pp_exact_r,pp_exact_f1 = zip(*result_pp)
    pp_shift_p,pp_shift_r,pp_shift_f1 = zip(*result_pp_shift)  
    print('Average testing F1 score with learning post-processing: ', np.average(pp_exact_f1))
    print('Average testing F1 score with zero parameter pp: ', np.average(nt_exact_f1))

    print('Average testing F1 score with learning post-processing allow shift: ', np.average(pp_shift_f1))
    print('Average testing F1 score with zero parameter pp allow shift: ', np.average(nt_shift_f1))

    print('Average testing precision with learning post-processing: ', np.average(pp_exact_p))
    print('Average testing precision with zero parameter pp: ', np.average(nt_exact_p))

    print('Average testing precision with learning post-processing allow shift: ', np.average(pp_shift_p))
    print('Average testing precision with zero parameter pp allow shift: ', np.average(nt_shift_p))

    print('Average testing recall with learning post-processing: ', np.average(pp_exact_r))
    print('Average testing recall with zero parameter pp : ', np.average(nt_exact_r))

    print('Average testing recall with learning post-processing allow shift: ', np.average(pp_shift_r))
    print('Average testing recall with zero parameter pp allow shift: ', np.average(nt_shift_r))

def all_test_only_e2e():
    contact_net.eval()
    lag_pp_net.eval()
    result_no_train = list()
    result_no_train_shift = list()
    result_pp = list()
    result_pp_shift = list()

    f1_no_train = list()
    f1_pp = list()
    seq_lens_list = list()

    # for long sequences
    batch_n = 0
    for seq_embedding_batch, PE_batch, _, comb_index, seq_embeddings, contacts, seq_lens in test_generator_1800:
        print('Batch number: ', batch_n)
        batch_n += 1

        state_pad = torch.zeros(1,2,2).to(device)
        seq_embedding_batch = seq_embedding_batch[0].to(device)
        PE_batch = PE_batch[0].to(device)
        seq_embedding = torch.Tensor(seq_embeddings.float()).to(device)
        contact_masks = torch.Tensor(contact_map_masks(seq_lens, 1800)).to(device)

        with torch.no_grad():
            pred_contacts = contact_net(PE_batch, seq_embedding_batch, state_pad)
            pred_u_map = combine_chunk_u_maps_no_replace(pred_contacts, comb_index, 6)
            pred_u_map = pred_u_map.unsqueeze(0)
            a_pred_list = lag_pp_net(pred_u_map, seq_embedding)

        # "Ground Truth" (actual labels of the test data)
        contacts_batch = torch.Tensor(contacts.float()[:,:1800, :1800])

        # the learning pp result
        final_pred = (a_pred_list[-1].cpu()>0.5).float()
        
        result_tmp       = list(map(lambda i: evaluate_exact   (final_pred.cpu()[i], contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        result_pp        += result_tmp

        result_tmp_shift = list(map(lambda i: evaluate_shifted (final_pred.cpu()[i], contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        result_pp_shift  += result_tmp_shift

        f1_tmp           = list(map(lambda i: F1_low_tri       (final_pred.cpu()[i], contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        f1_pp            += f1_tmp
        
        seq_lens_list += list(seq_lens)

    pp_exact_p,pp_exact_r,pp_exact_f1 = zip(*result_pp)
    pp_shift_p,pp_shift_r,pp_shift_f1 = zip(*result_pp_shift)  
    print('Average testing F1 score with learning post-processing: ', np.average(pp_exact_f1))
    print('Average testing F1 score with learning post-processing allow shift: ', np.average(pp_shift_f1))

    print('Average testing precision with learning post-processing: ', np.average(pp_exact_p))
    print('Average testing precision with learning post-processing allow shift: ', np.average(pp_shift_p))

    print('Average testing recall with learning post-processing: ', np.average(pp_exact_r))
    print('Average testing recall with learning post-processing allow shift: ', np.average(pp_shift_r))
    result_dict = dict()
    result_dict['exact_p'] = pp_exact_p
    result_dict['exact_r'] = pp_exact_r
    result_dict['exact_f1'] = pp_exact_f1
    result_dict['shift_p'] = pp_shift_p
    result_dict['shift_r'] = pp_shift_r
    result_dict['shift_f1'] = pp_shift_f1
    result_dict['seq_lens'] = seq_lens_list
    result_dict['exact_weighted_f1'] = np.sum(np.array(pp_exact_f1)*np.array(seq_lens_list)/np.sum(seq_lens_list))
    result_dict['shift_weighted_f1'] = np.sum(np.array(pp_shift_f1)*np.array(seq_lens_list)/np.sum(seq_lens_list))
    import _pickle as pickle
    with open('../results/rnastralign_long_e2e_evaluation_dict.pickle', 'wb') as f:
        pickle.dump(result_dict, f)

# There are three steps of training
# Last, joint fine tune
# final steps
def model_eval():
    contact_net.eval()
    lag_pp_net.eval()
    print('For short sequence:')
    contacts, seq_embeddings, matrix_reps, seq_lens = next(iter(val_generator_600))
    contacts_batch = torch.Tensor(contacts.float()).to(device)
    seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
    matrix_reps_batch = torch.unsqueeze(
        torch.Tensor(matrix_reps.float()).to(device), -1)

    # padding the states for supervised training with all 0s
    state_pad = torch.zeros(contacts.shape).to(device)
    PE_batch = get_pe(seq_lens, contacts.shape[-1]).float().to(device)
    with torch.no_grad():
        pred_contacts = contact_net(PE_batch, 
            seq_embedding_batch, state_pad)
        a_pred_list = lag_pp_net(pred_contacts, seq_embedding_batch)

    final_pred = (a_pred_list[-1].cpu()>0.5).float()

    result_tuple_list = list(map(lambda i: evaluate_exact(final_pred.cpu()[i], 
        contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
    exact_p,exact_r,exact_f1 = zip(*result_tuple_list)
    print('Average testing precision: ', np.average(exact_p))
    print('Average testing recall score: ', np.average(exact_r))
    print('Average testing f1 score: ', np.average(exact_f1))

    result_tuple_list_shift = list(map(lambda i: evaluate_shifted(final_pred.cpu()[i], 
        contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
    shift_p,shift_r,shift_f1 = zip(*result_tuple_list_shift)  
    print('Average testing precision allow shift: ', np.average(shift_p))
    print('Average testing recall score allow shift: ', np.average(shift_r))
    print('Average testing f1 score allow shift: ', np.average(shift_f1))

    print('For long sequence:')
    seq_embedding_batch, PE_batch, _, comb_index, seq_embeddings, contacts, seq_lens = next(iter(val_generator_1800))
    state_pad = torch.zeros(1,2,2).to(device)
    seq_embedding_batch = seq_embedding_batch[0].to(device)
    PE_batch = PE_batch[0].to(device)
    seq_embedding = torch.Tensor(seq_embeddings.float()).to(device)
    contact_masks = torch.Tensor(contact_map_masks(seq_lens, 1800)).to(device)

    with torch.no_grad():
        pred_contacts = contact_net(PE_batch, seq_embedding_batch, state_pad)
        pred_u_map = combine_chunk_u_maps_no_replace(pred_contacts, comb_index, 6)
        pred_u_map = pred_u_map.unsqueeze(0)
        a_pred_list = lag_pp_net(pred_u_map, seq_embedding)

    #  ground truth 
    contacts_batch = torch.Tensor(contacts.float()[:,:1800, :1800])
    # the learning pp result
    final_pred = (a_pred_list[-1].cpu()>0.5).float()
    result_tuple_list = list(map(lambda i: evaluate_exact(final_pred.cpu()[i], 
        contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
    exact_p,exact_r,exact_f1 = zip(*result_tuple_list)
    print('Average testing precision: ', np.average(exact_p))
    print('Average testing recall score: ', np.average(exact_r))
    print('Average testing f1 score: ', np.average(exact_f1))

    result_tuple_list_shift = list(map(lambda i: evaluate_shifted(final_pred.cpu()[i], 
        contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
    shift_p,shift_r,shift_f1 = zip(*result_tuple_list_shift)  
    print('Average testing precision allow shift: ', np.average(shift_p))
    print('Average testing recall score allow shift: ', np.average(shift_r))
    print('Average testing f1 score allow shift: ', np.average(shift_f1))

if not args.test:
    all_optimizer.zero_grad()
    for epoch in range(epoches_third):
        rna_ss_e2e.train()
        all_optimizer.zero_grad()
        print('On short sequence phase:')
        for contacts, seq_embeddings, matrix_reps, seq_lens in train_generator_600:
            contacts_batch = torch.Tensor(contacts.float()).to(device)
            seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)

            contact_masks = torch.Tensor(contact_map_masks(seq_lens, 600)).to(device)
            # padding the states for supervised training with all 0s
            state_pad = torch.zeros(1,2,2).to(device)

            PE_batch = get_pe(seq_lens, 600).float().to(device)
            # the end to end model
            pred_contacts, a_pred_list = rna_ss_e2e(PE_batch, 
                seq_embedding_batch, state_pad)

            loss_u = criterion_bce_weighted(pred_contacts*contact_masks, contacts_batch)

            # Compute loss, consider the intermidate output
            if pp_loss == "l2":
                loss_a = criterion_mse(
                    a_pred_list[-1]*contact_masks, contacts_batch)
                for i in range(pp_steps-1):
                    loss_a += np.power(step_gamma, pp_steps-1-i)*criterion_mse(
                        a_pred_list[i]*contact_masks, contacts_batch)
                mse_coeff = 1.0/(train_maxof_seq_len*pp_steps)

            if pp_loss == 'f1':
                loss_a = f1_loss(a_pred_list[-1]*contact_masks, contacts_batch)
                for i in range(pp_steps-1):
                    loss_a += np.power(step_gamma, pp_steps-1-i)*f1_loss(
                        a_pred_list[i]*contact_masks, contacts_batch)            
                mse_coeff = 1.0/pp_steps

            loss_a = mse_coeff*loss_a

            loss = loss_u + loss_a
            # print(steps_done)
            if steps_done % OUT_STEP ==0:
                print('Stage 3, epoch {}, step: {}, loss_u: {}, loss_a: {}, loss: {}'.format(
                    epoch, steps_done, loss_u, loss_a, loss))

                final_pred = a_pred_list[-1].cpu()>0.5
                f1 = list(map(lambda i: F1_low_tri(final_pred.cpu()[i], 
                    contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
                print('Average training F1 score: ', np.average(f1))
            # pdb.set_trace()

            # Optimize the model, we increase the batch size by 100 times
            loss.backward()
            if steps_done % 30 ==0:
                all_optimizer.step()
                all_optimizer.zero_grad()
            steps_done=steps_done+1
            if steps_done % 200 ==0:
                break

        print('On long sequence phase:')
        all_optimizer.zero_grad()
        for seq_embedding_batch, PE_batch, _, comb_index, seq_embeddings, contacts, seq_lens in train_generator_1800:
            state_pad = torch.zeros(1,2,2).to(device)
            seq_embedding_batch = seq_embedding_batch[0].to(device)
            PE_batch = PE_batch[0].to(device)
            contact = torch.Tensor(contacts.float()).to(device)
            seq_embedding = torch.Tensor(seq_embeddings.float()).to(device)
            contact_masks = torch.Tensor(contact_map_masks(seq_lens, 1800)).to(device)

            pred_contacts_first_batch = contact_net(PE_batch[:8, :, :], 
                seq_embedding_batch[:8, :, :], state_pad)
            pred_contacts_second_batch = contact_net(PE_batch[8:, :, :], 
                seq_embedding_batch[8:, :, :], state_pad)
            pred_contacts = torch.cat([pred_contacts_first_batch, pred_contacts_second_batch], 0)

            pred_u_map = combine_chunk_u_maps_no_replace(pred_contacts, comb_index, 6)
            pred_u_map = pred_u_map.unsqueeze(0)
            a_pred_list = lag_pp_net(pred_u_map, seq_embedding)

            # Compute loss
            loss_u = criterion_bce_weighted(pred_u_map*contact_masks, contact)

            # Compute loss, consider the intermidate output
            if pp_loss == "l2":
                loss_a = criterion_mse(
                    a_pred_list[-1]*contact_masks, contact)
                for i in range(pp_steps-1):
                    loss_a += np.power(step_gamma, pp_steps-1-i)*criterion_mse(
                        a_pred_list[i]*contact_masks, contact)
                mse_coeff = 1.0/(train_maxof_seq_len*pp_steps)

            if pp_loss == 'f1':
                loss_a = f1_loss(a_pred_list[-1]*contact_masks, contact)
                for i in range(pp_steps-1):
                    loss_a += np.power(step_gamma, pp_steps-1-i)*f1_loss(
                        a_pred_list[i]*contact_masks, contact)            
                mse_coeff = 1.0/pp_steps

            loss_a = mse_coeff*loss_a

            loss = loss_u + loss_a
            # print(steps_done)
            if steps_done % OUT_STEP ==0:
                print('Stage 3, epoch {}, step: {}, loss_u: {}, loss_a: {}, loss: {}'.format(
                    epoch, steps_done, loss_u, loss_a, loss))

                final_pred = a_pred_list[-1].cpu()>0.5
                f1 = list(map(lambda i: F1_low_tri(final_pred.cpu()[i], 
                    contact.cpu()[i]), range(contact.shape[0])))
                print('Average training F1 score: ', np.average(f1))
            # pdb.set_trace()

            # Optimize the model, we increase the batch size by 100 times
            loss.backward()
            if steps_done % 30 ==0:
                all_optimizer.step()
                all_optimizer.zero_grad()
            steps_done=steps_done+1
            if steps_done % 200 ==0:
                break

        if epoch%evaluate_epi==0:
            model_eval()
            torch.save(rna_ss_e2e.state_dict(), e2e_model_path)

all_test_only_e2e() 
# e2efold.evaluation.all_test_only_e2e(test_generator, contact_net, lag_pp_net, device, test_data, nameof_exper, cond_save_ct_predictions)