% Prepare motion-corrected 3D Spiral-QALAS HDF5 data for ZS-SSL.
% Supports the 3T (1-mm) and 7T (750-um) acquisitions.
% Requires vNav motion transforms: tform_matrix{arm}, one per acquired arm.

clear; close all; clc;

%% Settings
SCAN = '3T';                      % '3T' or '7T'
CC = 16;
NCONTRAST = 5;
NY_REF = 32;
NZ_REF = 32;
ESPIRIT_CROP = 0.5;
KEEP_TEMP = false;

if strcmp(SCAN,'3T')
    data_dir = '/autofs/space/daisy_002/users/unay/2026_09_23_bay5_invivo_spiral_qalas_vnav_motion_2';
    case_name = 'motion';           % 'motion' or 'nomotion'

    if strcmp(case_name,'motion')
        data_file = fullfile(data_dir,'meas_MID00085_FID41542_VNAV_SPIRAL_QALAS_MOTION_R1.dat');
    else
        data_file = fullfile(data_dir,'meas_MID00084_FID41541_VNAV_SPIRAL_QALAS_NOMOTION_R1.dat');
    end

    seq_file = fullfile(data_dir,'qalas_spiral_1mm_vds_slab_sel_TGA_R1_optgap_v7_nav_strictFA_prisma.seq');
    motion_dir = fullfile(data_dir,['motion_estimation_' case_name]);
    motion_file = fullfile(motion_dir,'_moco_1TR_motion.mat');
    compression_file = fullfile(motion_dir,'coil_compression.mat');
    pulseq_dir = '/autofs/cluster/berkin/unay/mysoftware/pulseq_v151';
    python_helper = fullfile(data_dir,'spiral_raw_io_moco.py');

    READOUT_ACCEL = 2;             % Retain every second spiral ADC sample
    ARM_ACCELERATIONS = [1];      % Can also use [1 2 4 8]
    USE_SAVED_COMPRESSION = true;
    tmp_dir = fullfile(data_dir,['tmp_zsssl_' case_name]);

else % 7T
    data_dir = '/autofs/space/daisy_002/users/unay/2026_10_23_bay2_invivo';
    data_file = fullfile(data_dir,'meas_MID00088_FID73895_pulseq151fix_spiralR2x1_Vref230_Vprep450.dat');
    seq_file = fullfile(data_dir,'qalas_spiral_750um_vds_slab_sel_TGA_R2x1_optgap_v8_nav_7T_TerraX.seq');
    motion_file = '';              % Set to 7T vNav motion-estimation .mat file
    compression_file = '';
    pulseq_dir = '/autofs/cluster/berkin/xingwang/Syncbb/others_toolboxes/pulseq_matlab_diff_versions/pulseq_v151/matlab';
    python_helper = '/autofs/space/daisy_002/users/unay/2026_09_23_bay5_invivo_spiral_qalas_vnav_motion_2/spiral_raw_io_moco.py';

    READOUT_ACCEL = 1;             % No retrospective ADC decimation
    ARM_ACCELERATIONS = 1;         % 7T sequence already has Rz=2
    USE_SAVED_COMPRESSION = false;
    tmp_dir = fullfile(data_dir,'tmp_7T_invivo_750um_R2x1_moco');
end

python_exe = '/autofs/space/marduk_003/users/unay/conda_envs/ssl_mrinufft/bin/python';
bart_dir = '/autofs/cluster/berkin/unay/mysoftware/bart-1.0.0-ardebil';
bart = fullfile(bart_dir,'bart');

restoredefaultpath;
addpath(genpath(pulseq_dir));
addpath(genpath('/cluster/berkin/berkin/Matlab_Code_New/LIBRARY/'));
addpath(fullfile(bart_dir,'matlab'));
setenv('CUDA_VISIBLE_DEVICES','0');
setenv('BART_GPU_GLOBAL_MEMORY','1');

assert(isfile(data_file),'Raw data not found: %s',data_file);
assert(isfile(seq_file),'Pulseq sequence not found: %s',seq_file);
assert(~isempty(motion_file) && isfile(motion_file), ...
    'Motion transforms not found. Set motion_file to the vNav tform_matrix MAT file.');
assert(isfile(python_helper),'Python HDF5 helper not found: %s',python_helper);
assert(isfile(bart),'BART not found: %s',bart);
if USE_SAVED_COMPRESSION
    assert(isfile(compression_file),'Saved coil compression not found: %s',compression_file);
end

if exist(tmp_dir,'dir'); rmdir(tmp_dir,'s'); end
mkdir(tmp_dir);

%% Read sequence and raw-data index
seq = mr.Sequence();
seq.read(seq_file);
FOV = double(seq.getDefinition('FOV')); FOV = FOV(:).';
MATRIX = double(seq.getDefinition('Matrix')); MATRIX = MATRIX(:).';
NX = MATRIX(1); NY = MATRIX(2); NZ = MATRIX(3);
NINTERL = round(double(seq.getDefinition('Ninterl')));
NZ_ACQ = round(double(seq.getDefinition('nETL')));
n_readout = round(double(seq.getDefinition('n_readout')));
NZ_NAV = round(double(seq.getDefinition('Nz_nav')));

assert(n_readout == NCONTRAST,'Unexpected number of QALAS contrasts.');
assert(all(mod(NINTERL,ARM_ACCELERATIONS)==0),'Invalid arm acceleration.');
fprintf('\n%s: Matrix [%d %d %d], acquired kz %d/%d, arms %d\n', ...
    SCAN,MATRIX,NZ_ACQ,NZ,NINTERL);

% Identify Cartesian reference, QALAS, and vNav readouts by their populations.
twix = mapVBVD(data_file);
if iscell(twix); twix = twix{end}; end
im = twix.image;
NCha = im.NCha;
u_center = unique(im.centerCol);
center_counts = zeros(size(u_center));
for ii = 1:numel(u_center)
    center_counts(ii) = sum(im.centerCol == u_center(ii));
end
expected_ref = NY_REF * NZ_REF;
expected_qalas = NINTERL * NCONTRAST * NZ_ACQ;
expected_nav = NINTERL * NZ_NAV;
centerCol_ref = u_center(center_counts == expected_ref);
centerCol_qalas = u_center(center_counts == expected_qalas);
centerCol_nav = u_center(center_counts == expected_nav);
assert(isscalar(centerCol_ref) && isscalar(centerCol_qalas) && isscalar(centerCol_nav), ...
    'Could not identify reference, QALAS, and vNav readouts.');
idx_ref = find(im.centerCol == centerCol_ref);
idx_qalas = find(im.centerCol == centerCol_qalas);
idx_nav = find(im.centerCol == centerCol_nav);

% Recover the ADC sample lengths from the exact Pulseq sequence.
be = vertcat(seq.blockEvents{:});
column_counts = sum(be ~= 0,1);
adc_col = find(column_counts == numel(im.memPos));
assert(isscalar(adc_col),'Could not identify the Pulseq ADC column.');
adc_ids = be(:,adc_col);
adc_ids = adc_ids(adc_ids ~= 0);
adc_sample_counts = zeros(numel(adc_ids),1);
for ii = 1:numel(adc_ids)
    adc_sample_counts(ii) = round(seq.adcLibrary.data(adc_ids(ii)).array(1));
end
REF_SAMPLES = [];
SPIRAL_SAMPLES = [];
VNAV_SAMPLES = [];
u_ns = unique(adc_sample_counts);
for ii = 1:numel(u_ns)
    n_events = sum(adc_sample_counts == u_ns(ii));
    if n_events == expected_ref
        REF_SAMPLES = u_ns(ii);
    elseif n_events == expected_qalas
        SPIRAL_SAMPLES = u_ns(ii);
    elseif n_events == expected_nav
        VNAV_SAMPLES = u_ns(ii);
    end
end
assert(~isempty(REF_SAMPLES) && ~isempty(SPIRAL_SAMPLES) && ~isempty(VNAV_SAMPLES), ...
    'Could not identify ADC sample lengths.');
assert(REF_SAMPLES == NX,'Reference sample count does not match NX.');

readout_indices = 1:READOUT_ACCEL:SPIRAL_SAMPLES;
N_READOUT_OUT = numel(readout_indices);
N_SAMPLES_PER_ARM = N_READOUT_OUT * NZ_ACQ;
voxel_m = FOV ./ MATRIX;
fprintf('Spiral samples/readout: %d -> %d, coils: %d -> %d\n', ...
    SPIRAL_SAMPLES,N_READOUT_OUT,NCha,CC);

%% Coil compression and ESPIRiT sensitivity maps
ref_raw = readraw(data_file,im.memPos(idx_ref),REF_SAMPLES,NCha,im.IsReflected(idx_ref));
ref_base = fullfile(tmp_dir,'qalas_ref');
sens_base = fullfile(tmp_dir,'qalas_sens');
iy = center_indices(NY,NY_REF);
iz = center_indices(NZ,NZ_REF);

if USE_SAVED_COMPRESSION
    S = load(compression_file,'coil_compression');
    coil_compression = S.coil_compression;
else
    % The 7T pipeline estimates coil compression from its own reference scan.
    ref_cc_grid = permute(reshape(ref_raw,[REF_SAMPLES NCha NY_REF NZ_REF]),[1 3 4 2]);
    ref_full = complex(zeros(NX,NY,NZ,NCha,'single'));
    ref_full(:,iy,iz,:) = ref_cc_grid;
    wcfl(ref_base,ref_full);
    cc_base = fullfile(tmp_dir,'cc_matrix');
    cmd = sprintf('"%s" cc -M -p %d -S "%s" "%s"',bart,CC,ref_base,cc_base);
    assert(system(cmd)==0,'BART coil compression failed.');
    C = squeeze(rcfl(cc_base));
    coil_compression = conj(C(:,1:CC));
    clear ref_cc_grid ref_full C
end
assert(isequal(size(coil_compression),[NCha CC]),'Coil compression dimension mismatch.');

ref_cc = apply_coil_compression(ref_raw,coil_compression);
ref_cc = permute(reshape(ref_cc,[REF_SAMPLES CC NY_REF NZ_REF]),[1 3 4 2]);
ref_full = complex(zeros(NX,NY,NZ,CC,'single'));
ref_full(:,iy,iz,:) = ref_cc;
wcfl(ref_base,ref_full);
cmd = sprintf('"%s" ecalib -m 1 -r 22 -c %.3f "%s" "%s"', ...
    bart,ESPIRIT_CROP,ref_base,sens_base);
assert(system(cmd)==0,'BART ESPIRiT calibration failed.');
sens_maps = squeeze(rcfl(sens_base));
assert(isequal(size(sens_maps),[NX NY NZ CC]),'Sensitivity map dimension mismatch.');

sens_slice = squeeze(sens_maps(:,:,round(NZ/2),:));
sens_support = sum(abs(sens_slice).^2,3) > 0;
figure('Name','ESPIRiT sensitivity support');
imagesc(rot90(sens_support)); axis image off; colormap gray;
title(sprintf('%s ESPIRiT support, crop = %.2f',SCAN,ESPIRIT_CROP));
clear ref_raw ref_cc ref_full sens_slice sens_support

%% Exact Pulseq trajectory and vNav motion transforms
warn_state = warning('query','mr:restoreShape');
warning('off','mr:restoreShape');
ktraj_adc = seq.calculateKspacePP();
warning(warn_state.state,'mr:restoreShape');

N_REF_SAMPLES_TOTAL = expected_ref * REF_SAMPLES;
N_QALAS_PER_CONTRAST = SPIRAL_SAMPLES * NZ_ACQ;
N_QALAS_PER_ARM = N_QALAS_PER_CONTRAST * NCONTRAST;
N_VNAV_PER_ARM = VNAV_SAMPLES * NZ_NAV;
expected_samples = N_REF_SAMPLES_TOTAL + NINTERL*(N_QALAS_PER_ARM+N_VNAV_PER_ARM);
assert(size(ktraj_adc,2)==expected_samples,'Pulseq ADC sample count mismatch.');

ktraj = zeros(3,N_READOUT_OUT,NZ_ACQ,NCONTRAST,NINTERL,'like',ktraj_adc);
cursor = N_REF_SAMPLES_TOTAL + 1;
for arm = 1:NINTERL
    for c = 1:NCONTRAST
        last_sample = cursor + N_QALAS_PER_CONTRAST - 1;
        traj_arm = reshape(ktraj_adc(:,cursor:last_sample),3,SPIRAL_SAMPLES,NZ_ACQ);
        ktraj(:,:,:,c,arm) = traj_arm(:,readout_indices,:);
        cursor = last_sample + 1;
    end
    cursor = cursor + N_VNAV_PER_ARM;
end
assert(cursor==size(ktraj_adc,2)+1,'Trajectory parsing mismatch.');
clear ktraj_adc

M = load(motion_file,'tform_matrix');
tform_matrix = M.tform_matrix;
assert(numel(tform_matrix)==NINTERL,'Expected one vNav transform per acquired arm.');
fprintf('Loaded %d vNav motion transforms.\n',numel(tform_matrix));

%% Motion-corrected HDF5 output
for ACCELERATION = ARM_ACCELERATIONS
    NINTERL_OUT = NINTERL/ACCELERATION;
    N_SAMPLES_3D = N_SAMPLES_PER_ARM*NINTERL_OUT;

    if strcmp(SCAN,'3T')
        out_file = fullfile(data_dir,sprintf( ...
            'spiral_qalas_%s_R%dx%d_mc_3d_moco_input.h5', ...
            case_name,READOUT_ACCEL,ACCELERATION));
    else
        out_file = fullfile(data_dir,'spiral_qalas_7T_invivo_750um_R2x1_mc_3d_moco_input.h5');
    end

    % Match the contrast-dependent arm order of the accelerated sequence.
    kept_interleaves = zeros(NCONTRAST,NINTERL_OUT);
    for c = 1:NCONTRAST
        if ACCELERATION == 8
            if ismember(c,[1 3 5])
                kept_interleaves(c,:) = 1:2:15;
            else
                kept_interleaves(c,:) = 64:-2:50;
            end
        else
            if ismember(c,[1 3 5])
                kept_interleaves(c,:) = 1:NINTERL_OUT;
            else
                kept_interleaves(c,:) = (NINTERL-NINTERL_OUT+1):NINTERL;
            end
        end
    end

    kspace_full = complex(zeros(NCONTRAST,CC,N_SAMPLES_3D,'single'));
    trajectory_full = zeros(NCONTRAST,N_SAMPLES_3D,3,'single');
    norm_before_sq = 0;
    norm_after_sq = 0;
    max_rot_norm_error = 0;

    for c = 1:NCONTRAST
        fprintf('%s, arm acceleration %d, contrast %d/%d\n',SCAN,ACCELERATION,c,NCONTRAST);
        for out_arm = 1:NINTERL_OUT
            arm = kept_interleaves(c,out_arm);
            first_qalas = (arm-1)*NCONTRAST*NZ_ACQ;
            ind_c = idx_qalas(first_qalas+(c-1)*NZ_ACQ+(1:NZ_ACQ));
            raw = readraw(data_file,im.memPos(ind_c), ...
                SPIRAL_SAMPLES,NCha,im.IsReflected(ind_c));
            raw_cc = apply_coil_compression(raw,coil_compression);
            raw_cc = raw_cc(readout_indices,:,:);
            raw_matrix = reshape(permute(raw_cc,[1 3 2]),N_SAMPLES_PER_ARM,CC);
            traj_phys = reshape(ktraj(:,:,:,c,arm),3,N_SAMPLES_PER_ARM);
            k_nom = double(traj_phys).*voxel_m(:);

            % Apply the same samplewise rotation and translation correction at 3T and 7T.
            [k_use,raw_use] = correct_spiral_kspace_sample_motion( ...
                k_nom,raw_matrix,tform_matrix{arm},MATRIX);

            norm_error = max(abs(vecnorm(double(k_use),2,1)-vecnorm(double(k_nom),2,1)));
            max_rot_norm_error = max(max_rot_norm_error,norm_error);
            start_out = (out_arm-1)*N_SAMPLES_PER_ARM+1;
            stop_out = out_arm*N_SAMPLES_PER_ARM;
            kspace_full(c,:,start_out:stop_out) = permute(raw_use,[3 2 1]);
            trajectory_full(c,start_out:stop_out,:) = ...
                reshape(single(k_use.'),1,N_SAMPLES_PER_ARM,3);
            norm_before_sq = norm_before_sq+double(sum(abs(raw_matrix(:)).^2));
            norm_after_sq = norm_after_sq+double(sum(abs(raw_use(:)).^2));
        end
    end

    fprintf('MoCo norm relative change: %.3e; trajectory norm error: %.3e\n', ...
        abs(sqrt(norm_after_sq)-sqrt(norm_before_sq))/sqrt(norm_before_sq),max_rot_norm_error);
    assert(all(isfinite(kspace_full(:))) && all(isfinite(trajectory_full(:))), ...
        'Non-finite motion-corrected measurements.');

    kspace_file = fullfile(tmp_dir,'kspace_precomputed.bin');
    traj_file = fullfile(tmp_dir,'trajectory_precomputed.bin');
    sens_file = fullfile(tmp_dir,'sens_maps_precomputed.bin');
    write_complex_bin(kspace_file,kspace_full);
    fid = fopen(traj_file,'wb'); assert(fid~=-1);
    fwrite(fid,trajectory_full,'single'); fclose(fid);
    write_complex_bin(sens_file,sens_maps);

    cmd = sprintf(['"%s" "%s" write_h5_precomputed ' ...
        '"%s" "%s" "%s" "%s" %d %d %d %d %d %d'], ...
        python_exe,python_helper,out_file,kspace_file,traj_file,sens_file, ...
        NCONTRAST,CC,N_SAMPLES_3D,NX,NY,NZ);
    assert(system(cmd)==0,'Python HDF5 packaging failed.');
    assert(isfile(out_file),'HDF5 was not created.');

    kinfo = h5info(out_file,'/kspace');
    tinfo = h5info(out_file,'/trajectory');
    sinfo = h5info(out_file,'/sens_maps');
    kshape = fliplr(kinfo.Dataspace.Size);
    tshape = fliplr(tinfo.Dataspace.Size);
    sshape = fliplr(sinfo.Dataspace.Size);
    assert(isequal(kshape,[NCONTRAST CC N_SAMPLES_3D]));
    assert(isequal(tshape,[NCONTRAST N_SAMPLES_3D 3]));
    assert(isequal(sshape,[NX NY NZ CC]));
    fprintf('Saved motion-corrected HDF5: %s\n',out_file);
    clear kspace_full trajectory_full raw raw_cc raw_matrix
end

if ~KEEP_TEMP; rmdir(tmp_dir,'s'); end

%% Local functions
function [k_cor,raw_corr] = correct_spiral_kspace_sample_motion(k_nom,raw_cc,tform_tr,Nseq)
    % Rotation is applied to k-space; translation is applied as a phase ramp.
    P = [0 1 0; 1 0 0; 0 0 1];
    R_world = double(tform_tr.R);
    t_world = double(tform_tr.Translation(:));
    R_array = P.'*R_world*P;
    t_array = P.'*t_world;
    center_array = [double(Nseq(1))/2+1; double(Nseq(2))/2+1; double(Nseq(3))/2+1];
    t_eff = t_array+(R_array-eye(3))*center_array;
    k_cor = R_array.'*double(k_nom);
    phase_corr = exp(1i*2*pi*(t_eff.'*double(k_nom)));
    raw_corr = raw_cc.*phase_corr.';
    k_cor = single(k_cor);
    raw_corr = single(raw_corr);
end

function ind = center_indices(N,n)
    start_ind = floor((N-n)/2)+1;
    ind = start_ind:(start_ind+n-1);
end

function data_cc = apply_coil_compression(data,W)
    sz = size(data);
    x = permute(data,[1 3 2]);
    x = reshape(x,[],sz(2));
    x = x*W;
    data_cc = reshape(x,sz(1),sz(3),size(W,2));
    data_cc = permute(data_cc,[1 3 2]);
end

function data = readraw(file,pos,NCol,NCha,refl)
    fid = fopen(file,'r');
    assert(fid~=-1,'Could not open raw data.');
    data = complex(zeros(NCol,NCha,numel(pos),'single'));
    for i = 1:numel(pos)
        fseek(fid,double(pos(i))+192,'bof');
        x = fread(fid,[2 (NCol+4)*NCha],'float32=>single');
        x = complex(x(1,:),x(2,:));
        x = reshape(x,[NCol+4 NCha]);
        x = x(5:end,:);
        if refl(i); x = flip(x,1); end
        data(:,:,i) = x;
    end
    fclose(fid);
end

function write_complex_bin(filename,x)
    x = single(x(:));
    raw = zeros(2*numel(x),1,'single');
    raw(1:2:end) = real(x);
    raw(2:2:end) = imag(x);
    fid = fopen(filename,'wb');
    assert(fid~=-1,'Could not open binary output.');
    fwrite(fid,raw,'single');
    fclose(fid);
end

function wcfl(name,x)
    d = size(x);
    d = [d ones(1,16-numel(d))];
    fid = fopen([name '.hdr'],'w');
    fprintf(fid,'# Dimensions\n');
    fprintf(fid,'%d ',d);
    fclose(fid);
    x = single(x(:));
    fid = fopen([name '.cfl'],'w');
    fwrite(fid,[real(x).';imag(x).'],'float32');
    fclose(fid);
end

function x = rcfl(name)
    fid = fopen([name '.hdr']);
    fgetl(fid);
    d = sscanf(fgetl(fid),'%d').';
    fclose(fid);
    last = find(d~=1,1,'last');
    if isempty(last); d = 1; else; d = d(1:last); end
    fid = fopen([name '.cfl']);
    a = fread(fid,[2 prod(d)],'float32=>single');
    fclose(fid);
    x = reshape(complex(a(1,:),a(2,:)),d);
end
