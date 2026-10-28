import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='ZS-SSL: Zero-Shot Self-Supervised Learning')

    # %% hyperparameters for the  network
    parser.add_argument('--data_opt', type=str, default='Coronal_PD',
                    help='type of dataset')
    parser.add_argument('--data_dir', type=str, default='/home/zs_ssl/data.mat',
                    help='data directory')                
    parser.add_argument('--nrow_GLOB', type=int, default=320,
                        help='number of rows of the slices in the dataset')
    parser.add_argument('--ncol_GLOB', type=int, default=368,
                        help='number of columns of the slices in the dataset')
    parser.add_argument('--ncoil_GLOB', type=int, default=15,
                        help='number of coils of the slices in the dataset')                
    parser.add_argument('--acc_rate', type=int, default=4,
                        help='acceleration rate')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--batchSize', type=int, default=1,
                        help='batch size')
    parser.add_argument('--nb_unroll_blocks', type=int, default=10,
                        help='number of unrolled blocks')
    parser.add_argument('--nb_res_blocks', type=int, default=15,
                        help="number of residual blocks in ResNet")
    parser.add_argument('--nb_features', type=int, default=8,
                        help="number of feature maps in ResNet (default: 8)")
    parser.add_argument('--lambda_start', type=float, default=0.1,
                        help="initial value of the learned regularization parameter lambda (default: 0.1)")
    parser.add_argument('--lr_min', type=float, default=0.0,
                        help="minimum learning rate for cosine annealing schedule (default: 0, i.e. no schedule)")
    parser.add_argument('--CG_Iter', type=int, default=10,
                        help='number of Conjugate Gradient iterations for DC')
    parser.add_argument('--CG_tol', type=float, default=1e-4, help="Relative residual tolerance for CG stopping.")
    parser.add_argument('--cg_coil_batch', type=int, default=0,
                        help="Coils per FFT chunk in fast batched-contrast Toeplitz CG. 0 uses the existing operator n_trans (3D currently 1).")
    parser.add_argument(
        "--implicit_cg",
        action="store_true",
        help="Use implicit differentiation through the CG solve.",
    )

    # %% hyperparameters for the zs-ssl
    parser.add_argument('--rho_val', type=float, default=0.2,
                        help='cardinality of the validation mask')                        
    parser.add_argument('--rho_train', type=float, default=0.4,
                        help='cardinality of the loss mask, \ rho = |\ Lambda| / |\ Omega|')
    parser.add_argument('--num_reps', type=int, default=25,
                        help='number of repetions for the remainder mask')
    parser.add_argument('--transfer_learning', type=bool, default=False,
                        help='transfer learning from pretrained model')
    parser.add_argument('--TL_path', type=str, default=None,
                        help='path to pretrained model')                                        
    parser.add_argument('--stop_training', type=int, default=25,
                        help='stop training if a new lowest validation loss hasnt been achieved in xx epochs')                    

    parser.add_argument('--saved_model_name', type=str, default='ZS_SSL_Model_300Epochs_Rate4_10Unrolls',
                        help='model name to be used for eval')      
    parser.add_argument(
        '--progressive_cg',
        action='store_true',
        help='Automatically distribute CG tolerances across unroll blocks.'
    ) 
    parser.add_argument(
        '--cg_diagnostics',
        action='store_true',
        help='Print CG iteration statistics per unroll block.'
    )
    parser.add_argument(
        '--backward_cg_warmstart',
        action='store_true',
        help='Warm-start backward CG using the solution from the next unroll block.'
    )
    return parser