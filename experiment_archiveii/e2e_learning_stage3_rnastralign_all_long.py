from torch.utils import data

from e2efold.models import ContactNetwork, ContactNetwork_test, ContactNetwork_fc
from e2efold.models import ContactAttention, ContactAttention_simple_fix_PE
from e2efold.models import Lag_PP_NN, RNA_SS_e2e, Lag_PP_zero, Lag_PP_perturb, Lag_PP_final
from e2efold.models import Lag_PP_mixed, ContactAttention_simple
from e2efold.common.utils import *
from e2efold.common.config import process_config
from e2efold.evaluation import all_test_only_e2e
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

os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_torch(0)
# END Config Setup

# START Data Loading
from e2efold.data_generator import RNASSDataGenerator, Dataset, Dataset_1800
import collections
RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq ss_label length name pairs')

train_data_600 = RNASSDataGenerator(config.data_root+'data/{}/'.format(data_type), 'train_600') # indicating that the model was trainied on this dataset
test_data_1800 = RNASSDataGenerator(config.data_root+'data/{}/'.format(config.test_data_type), 'all_1800')

train_maxof_seq_len = train_data_600.maxof_seq_len # this value is used as a part of the model training algrorithm, currently 600, needed as the L param for instantiating the contact_net below
print(f'train_maxof_seq_len: {train_maxof_seq_len}')

test_maxof_seq_len = test_data_1800.maxof_seq_len
print(f'test_maxof_seq_len: {test_maxof_seq_len}')

params              = {'batch_size': 1, 'shuffle': False, 'num_workers': 6, 'drop_last': False}
test_set_1800       = Dataset_1800(test_data_1800)
test_generator_1800 = data.DataLoader(test_set_1800, **params)

if (model_type == 'test_lc'):        contact_net = ContactNetwork_test           (d=d, L=train_maxof_seq_len).to(device)
if (model_type == 'att6'):           contact_net = ContactAttention              (d=d, L=train_maxof_seq_len).to(device)
if (model_type == 'att_simple'):     contact_net = ContactAttention_simple       (d=d, L=train_maxof_seq_len).to(device)    
if (model_type == 'att_simple_fix'): contact_net = ContactAttention_simple_fix_PE(d=d, L=train_maxof_seq_len).to(device)
if (model_type == 'fc'):             contact_net = ContactNetwork_fc             (d=d, L=train_maxof_seq_len).to(device)
if (model_type == 'conv2d_fc'):      contact_net = ContactNetwork                (d=d, L=train_maxof_seq_len).to(device)

# contact_net.conv1d2.register_forward_hook(get_activation('conv1d2')). this is for debugging

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

# END Data Loading

#  dif between these two methods (model_eval_all_test() and all_test_only_e2e()) asking ChatGPT:
'''
model_eval_all_test():
comprehensive evaluation of the model on the test dataset.
Evaluates the model with and without post-processing (PP).

all_test_only_e2e():
evaluating the model with end-to-end (E2E) predictions.
Evaluates the model only with post-processing.
'''
def model_eval_all_test():
    contact_net.eval()
    lag_pp_net.eval()
    result_no_train = list()
    result_no_train_shift = list()
    result_pp = list()
    result_pp_shift = list()

    f1_no_train = list()
    f1_pp = list()
    # for long sequences
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
            pred_u_map = combine_chunk_u_maps_no_replace(pred_contacts, comb_index, 6).unsqueeze(0)
            a_pred_list = lag_pp_net(pred_u_map, seq_embedding)

        #  ground truth 
        contacts_batch = torch.Tensor(contacts.float()[:,:1800, :1800])

        # only post-processing, with zero parameters
        u_no_train = postprocess(pred_u_map, seq_embedding, 0.01, 0.1, 50, 1.0, True)
        map_no_train = (u_no_train > 0.5).float()
        
        result_no_train_tmp = list(map(lambda i: evaluate_exact(map_no_train.cpu()[i], contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        result_no_train += result_no_train_tmp
        
        result_no_train_tmp_shift = list(map(lambda i: evaluate_shifted(map_no_train.cpu()[i], contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        result_no_train_shift += result_no_train_tmp_shift

        f1_no_train_tmp = list(map(lambda i: F1_low_tri(map_no_train.cpu()[i], contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        f1_no_train += f1_no_train_tmp

        # the learning pp result
        final_pred = (a_pred_list[-1].cpu()>0.5).float()
        result_tmp = list(map(lambda i: evaluate_exact(final_pred.cpu()[i], contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        result_pp += result_tmp

        result_tmp_shift = list(map(lambda i: evaluate_shifted(final_pred.cpu()[i], contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        result_pp_shift += result_tmp_shift

        f1_tmp = list(map(lambda i: F1_low_tri(final_pred.cpu()[i], contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        f1_pp += f1_tmp

    # nt = no train
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

    result_pp = list()
    result_pp_shift = list()

    f1_pp = list()
    seq_lens_list = list()

    countof_batches_elapsed = 0
    countof_seq_per_batch = test_generator_1800.batch_size
    
    for seq_embedding_batch, PE_batch, _, comb_index, seq_embeddings, contacts, seq_lens in test_generator_1800:
        if countof_batches_elapsed % 1==0:
            print('Batch number: ', countof_batches_elapsed)
        countof_batches_elapsed += 1

        # # PE = Position Embedding
        # state_pad           = torch.zeros(1,2,2)                   .to(device) # torch.Size([1, 2, 2])
        # PE_batch            = PE_batch[0]                          .to(device) # torch.Size([1, 15, 600, 4]) -> torch.Size([15, 600, 4])
        # seq_embedding_batch = seq_embedding_batch[0]               .to(device) # torch.Size([1, 15, 600, 4]) -> torch.Size([15, 600, 4])
        # seq_embedding       = torch.Tensor(seq_embeddings.float()) .to(device) # torch.Size([1, 1800, 4]) -> torch.Size([1, 1800, 4])
    
        state_pad = torch.zeros(1,2,2).to(device)
        seq_embedding_batch = seq_embedding_batch[0].to(device)
        PE_batch = PE_batch[0].to(device)
        seq_embedding = torch.Tensor(seq_embeddings.float()).to(device)
        contact_masks = torch.Tensor(contact_map_masks(seq_lens, 1800)).to(device)
    
        with torch.no_grad():
            
            # PE_600        = get_pe(seq_lens, contacts.shape[-1]).float() .to(device) # torch.Size([1, 1800, 111])
            # state_pad_600 = torch.zeros(contacts.shape)                  .to(device) # torch.Size([1, 1800, 1800])
            # pred_contacts_600 = contact_net(PE_600, seq_embedding, state_pad_600)
            # a_pred_list_600 = lag_pp_net(pred_contacts_600, seq_embedding_batch)
            
            pred_contacts = contact_net(PE_batch, seq_embedding_batch, state_pad)
            pred_u_map = combine_chunk_u_maps_no_replace(pred_contacts, comb_index, 6)
            pred_u_map = pred_u_map.unsqueeze(0)
            a_pred_list = lag_pp_net(pred_u_map, seq_embedding)

        # "Ground Truth" (actual labels of the test data)
        contacts_batch = torch.Tensor(contacts.float()[:,:1800, :1800])

        # Evalute Prediction of: U-Net + training of the PP-Net 
        # The return from evaluate_exact(...) is a tuple of 3 tensors, representing:
        # precision, recall, f1_score
        final_pred       = (a_pred_list[-1].cpu()>0.5).float()
        
        result_tmp       = list(map(lambda i: evaluate_exact   (final_pred.cpu()[i], contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        result_pp        += result_tmp

        result_tmp_shift = list(map(lambda i: evaluate_shifted (final_pred.cpu()[i], contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        result_pp_shift  += result_tmp_shift

        f1_tmp           = list(map(lambda i: F1_low_tri       (final_pred.cpu()[i], contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        f1_pp            += f1_tmp
        
        seq_lens_list += list(seq_lens)

    pp_exact_p,pp_exact_r,pp_exact_f1 = zip(*result_pp)
    pp_shift_p,pp_shift_r,pp_shift_f1 = zip(*result_pp_shift)  
    
    print('Number of sequences: ', len(pp_exact_f1))
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
    result_dict['name'] = [a.name for a in test_data_1800.data]
    # import _pickle as pickle
    # with open('../results/archiveii_long_e2e_evaluation_dict.pickle', 'wb') as f:
    #     pickle.dump(result_dict, f)

all_test_only_e2e()
# all_test_only_e2e(test_generator, contact_net, lag_pp_net, device, test_data, nameof_exper, cond_save_ct_predictions)