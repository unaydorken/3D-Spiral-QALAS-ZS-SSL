%% 01_estimate_motion_from_vnavs_nomotion.m

clear; close all; clc;

%% Paths

addpath(genpath('/autofs/cluster/berkin/yohan/matlab_code/pulseq-v1_5_1'));
addpath(genpath('/cluster/berkin/berkin/Matlab_Code_New/LIBRARY/'));

data_dir = ...
'/autofs/space/daisy_002/users/unay/2026_09_23_bay5_invivo_spiral_qalas_vnav_motion_2';

case_name = 'motion';
% case_name = 'motion';

switch lower(case_name)

    case 'nomotion'

        dat_file = fullfile(data_dir, ...
            'meas_MID00084_FID41541_VNAV_SPIRAL_QALAS_NOMOTION_R1.dat');

        out_dir = fullfile( ...
            data_dir, ...
            'motion_estimation_nomotion');

        case_label = 'No-motion';


    case 'motion'

        dat_file = fullfile(data_dir, ...
            'meas_MID00085_FID41542_VNAV_SPIRAL_QALAS_MOTION_R1.dat');

        out_dir = fullfile( ...
            data_dir, ...
            'motion_estimation_motion');

        case_label = 'Motion';


    otherwise

        error('Unknown case_name: %s',case_name);

end


seq_file = fullfile(data_dir, ...
'qalas_spiral_1mm_vds_slab_sel_TGA_R1_optgap_v7_nav_strictFA_prisma.seq');


bart = ...
'/autofs/cluster/berkin/unay/mysoftware/bart-1.0.0-ardebil/bart';


if ~exist(out_dir,'dir')
    mkdir(out_dir);
end

cd(out_dir);

fprintf('\n============================================================\n');
fprintf('Processing case: %s\n',case_label);
fprintf('Data: %s\n',dat_file);
fprintf('Output: %s\n',out_dir);
fprintf('============================================================\n');


%% Acquisition parameters

FOV = [0.220 0.220 0.144];

N_nav = [36 36 24];
Nz_nav = N_nav(3);

N_qalas = [220 220 144];

centerCol_ref   = 111;
centerCol_qalas = 1206;
centerCol_nav   = 4006;

NCol_ref   = 220;
NCol_qalas = 2410;
NCol_nav   = 8010;

NCC = 16;


%% Read TWIX

fprintf('\nReading TWIX index...\n');

twix = mapVBVD(dat_file);

if iscell(twix)
    twix = twix{end};
end

im = twix.image;

NCha = im.NCha;

idx_ref   = find(im.centerCol == centerCol_ref);
idx_qalas = find(im.centerCol == centerCol_qalas);
idx_nav   = find(im.centerCol == centerCol_nav);

fprintf('\n');
fprintf('Reference readouts : %d\n',numel(idx_ref));
fprintf('QALAS readouts     : %d\n',numel(idx_qalas));
fprintf('vNav readouts      : %d\n',numel(idx_nav));
fprintf('Physical coils     : %d\n',NCha);
fprintf('Virtual coils      : %d\n',NCC);

assert(numel(idx_ref)   == 1024);
assert(numel(idx_qalas) == 46080);
assert(numel(idx_nav)   == 1536);

n_vnav = numel(idx_nav) / Nz_nav;

assert(mod(numel(idx_nav),Nz_nav) == 0);
assert(n_vnav == 64);

fprintf('vNav volumes       : %d\n\n',n_vnav);



%% Read Cartesian reference

fprintf('\nReading Cartesian reference scan...\n');

ref_raw = readraw( ...
    dat_file, ...
    im.memPos(idx_ref), ...
    NCol_ref, ...
    NCha, ...
    im.IsReflected(idx_ref));

%% Compute coil compression for THIS scan

fprintf('\nComputing coil compression: %d -> %d\n',NCha,NCC);

X = permute(ref_raw,[1 3 2]);
X = reshape(X,[],NCha);

C = X' * X;

[V,D] = eig(C,'vector');

D = real(D);

[D_sorted,order] = sort(D,'descend');

V = V(:,order);

coil_compression = single(V(:,1:NCC));

energy_retained = ...
    sum(D_sorted(1:NCC)) / sum(D_sorted);

fprintf('Signal energy retained: %.2f %%\n', ...
    100*energy_retained);

clear X C V D D_sorted order;

% Apply same compression used for the high-resolution reconstruction

ref_raw = apply_coil_compression( ...
    ref_raw, ...
    coil_compression);

fprintf('Compressed reference size: [%s]\n', ...
    num2str(size(ref_raw)));


%% Reference -> vNav sensitivity maps

ref = reshape( ...
    ref_raw, ...
    [NCol_ref NCC 32 32]);

ref = permute(ref,[1 3 4 2]);

ix = center_indices(220,36);
iz = center_indices(32,24);

ref_crop = ref(ix,:,iz,:);

ref_vnav = complex(zeros( ...
    36,36,24,NCC,'single'));

iy = center_indices(36,32);

ref_vnav(:,iy,:,:) = ref_crop;

fprintf('\nReference for vNav sensitivity maps: [%s]\n', ...
    num2str(size(ref_vnav)));

wcfl('vnav_ref',ref_vnav);

fprintf('Estimating ESPIRiT sensitivity maps...\n');

cmd = sprintf( ...
    '"%s" ecalib -m 1 vnav_ref vnav_sens', ...
    bart);

fprintf('%s\n',cmd);

if system(cmd)
    error('BART ecalib failed.');
end

fprintf('Sensitivity maps complete.\n\n');


% Keep a copy in the motion-estimation folder
save( ...
    fullfile(out_dir,'coil_compression.mat'), ...
    'coil_compression', ...
    'NCha', ...
    'NCC', ...
    'energy_retained');


%% Read full Pulseq sequence

fprintf('Reading Pulseq sequence...\n');

seq = mr.Sequence();
seq.read(seq_file);


%% Calculate exact ADC trajectories

fprintf('Calculating ADC trajectories...\n');

warn_state = warning('query','mr:restoreShape');
warning('off','mr:restoreShape');

ktraj_adc = seq.calculateKspacePP();

warning(warn_state.state,'mr:restoreShape');

fprintf('Total ADC trajectory samples: %d\n', ...
    size(ktraj_adc,2));


%% Verify Pulseq ADC structure

fprintf('Verifying Pulseq ADC structure...\n');

be = vertcat(seq.blockEvents{:});

fprintf('Pulseq blocks: %d\n',size(be,1));

column_counts = sum(be ~= 0,1);

adc_col = find(column_counts == numel(im.memPos));

assert(isscalar(adc_col), ...
    'Could not uniquely identify ADC column.');

fprintf('ADC blockEvents column: %d\n',adc_col);

adc_ids = be(:,adc_col);
adc_ids = adc_ids(adc_ids ~= 0);

fprintf('Pulseq ADC events: %d\n',numel(adc_ids));

assert(numel(adc_ids) == numel(im.memPos), ...
    'Pulseq/TWIX ADC event count mismatch.');


%% Verify ADC sample populations

adc_sample_counts = zeros(numel(adc_ids),1);

for ii = 1:numel(adc_ids)

    adc_sample_counts(ii) = round( ...
        seq.adcLibrary.data(adc_ids(ii)).array(1));

end

fprintf('\nADC sample-count populations:\n');

u_ns = unique(adc_sample_counts);

for ii = 1:numel(u_ns)

    fprintf( ...
        '  %5d samples : %6d ADC events\n', ...
        u_ns(ii), ...
        sum(adc_sample_counts == u_ns(ii)));

end

assert(sum(adc_sample_counts == NCol_ref) == numel(idx_ref));
assert(sum(adc_sample_counts == NCol_qalas) == numel(idx_qalas));
assert(sum(adc_sample_counts == NCol_nav) == numel(idx_nav));


%% Extract vNav trajectories

fprintf('\nExtracting vNav trajectories...\n');

nav_traj = zeros( ...
    3, ...
    NCol_nav, ...
    Nz_nav, ...
    n_vnav, ...
    'single');

adc_pointer = 1;
nav_counter = 0;

for ib = 1:size(be,1)

    adc_id = be(ib,adc_col);

    if adc_id == 0
        continue;
    end

    ns = round( ...
        seq.adcLibrary.data(adc_id).array(1));

    ind = adc_pointer:(adc_pointer + ns - 1);

    if ns == NCol_nav

        nav_counter = nav_counter + 1;

        nav_number = ...
            ceil(nav_counter / Nz_nav);

        nav_partition = ...
            mod(nav_counter-1,Nz_nav) + 1;

        nav_traj(:,:,nav_partition,nav_number) = ...
            single(ktraj_adc(:,ind));

    end

    adc_pointer = adc_pointer + ns;

end

fprintf('ADC trajectory samples consumed: %d\n',adc_pointer-1);
fprintf('calculateKspacePP samples:        %d\n',size(ktraj_adc,2));
fprintf('vNav ADC events found:            %d\n',nav_counter);

assert(adc_pointer-1 == size(ktraj_adc,2), ...
    'ADC trajectory indexing mismatch.');

assert(nav_counter == numel(idx_nav), ...
    'Expected %d vNav ADCs, found %d.', ...
    numel(idx_nav), ...
    nav_counter);

clear ktraj_adc;


%% Convert Pulseq trajectory -> BART units

nav_traj(1,:,:,:) = ...
    nav_traj(1,:,:,:) * FOV(1);

nav_traj(2,:,:,:) = ...
    nav_traj(2,:,:,:) * FOV(2);

nav_traj(3,:,:,:) = ...
    nav_traj(3,:,:,:) * FOV(3);


fprintf('\nTrajectory ranges:\n');

fprintf('kx: %.3f to %.3f\n', ...
    min(nav_traj(1,:),[],'all'), ...
    max(nav_traj(1,:),[],'all'));

fprintf('ky: %.3f to %.3f\n', ...
    min(nav_traj(2,:),[],'all'), ...
    max(nav_traj(2,:),[],'all'));

fprintf('kz: %.3f to %.3f\n\n', ...
    min(nav_traj(3,:),[],'all'), ...
    max(nav_traj(3,:),[],'all'));


%% Reconstruct all 64 vNavs

fprintf( ...
    'Reconstructing %d vNav volumes using %d virtual coils...\n\n', ...
    n_vnav,NCC);

vnav = complex(zeros( ...
    N_nav(1), ...
    N_nav(2), ...
    N_nav(3), ...
    n_vnav, ...
    'single'));


for n = 1:n_vnav

    ind = idx_nav( ...
        (n-1)*Nz_nav + (1:Nz_nav));

    ksp = readraw( ...
        dat_file, ...
        im.memPos(ind), ...
        NCol_nav, ...
        NCha, ...
        im.IsReflected(ind));

    ksp = apply_coil_compression( ...
        ksp, ...
        coil_compression);

    % BART dimensions:
    % [1 samples kz coils]

    ksp = reshape( ...
        permute(ksp,[1 3 2]), ...
        [1 NCol_nav Nz_nav NCC]);

    traj = nav_traj(:,:,:,n);

    wcfl('vnav_ksp',ksp);
    wcfl('vnav_traj',traj);

    cmd = sprintf([ ...
        '"%s" pics -g -e -t vnav_traj ' ...
        '-R W:7:0:5e-4 -i 100 ' ...
        'vnav_ksp vnav_sens vnav_reco'], ...
        bart);

    if system(cmd)
        error('BART PICS failed at vNav %d.',n);
    end

    vnav(:,:,:,n) = ...
        squeeze(rcfl('vnav_reco'));

    fprintf( ...
        'vNav %2d / %2d reconstructed\n', ...
        n,n_vnav);

end


%% Save reconstructed vNavs

save( ...
    fullfile(out_dir,'vnav_recons.mat'), ...
    'vnav', ...
    'nav_traj', ...
    'coil_compression', ...
    'NCha', ...
    'NCC', ...
    'energy_retained', ...
    'FOV', ...
    'N_nav', ...
    '-v7.3');


%% vNav montage

z = round(Nz_nav/2);

M = zeros( ...
    8*N_nav(1), ...
    8*N_nav(2), ...
    'single');

for n = 1:n_vnav

    x = abs(vnav(:,:,z,n));

    if max(x(:)) > 0
        x = x / max(x(:));
    end

    x = rot90(x,-1);

    r = floor((n-1)/8);
    c = mod(n-1,8);

    M( ...
        r*N_nav(1)+(1:N_nav(1)), ...
        c*N_nav(2)+(1:N_nav(2))) = x;

end

figure('Color','w');

imagesc(M,[0 1]);
axis image off;
colormap gray;

title(sprintf( ...
    'No-motion scan: all %d vNavs, axial z = %d', ...
    n_vnav,z));

hold on;

for n = 1:n_vnav

    r = floor((n-1)/8);
    c = mod(n-1,8);

    text( ...
        c*N_nav(2)+2, ...
        r*N_nav(1)+2, ...
        num2str(n), ...
        'Color','w', ...
        'FontSize',7, ...
        'FontWeight','bold', ...
        'VerticalAlignment','top');

end

hold off;

exportgraphics( ...
    gcf, ...
    fullfile(out_dir,'vnav_montage.png'), ...
    'Resolution',200);


%% Motion estimation from dedicated vNavs

fprintf('\nEstimating motion from dedicated vNavs...\n');

[optimizer, metric] = imregconfig('monomodal');
transformType = 'rigid';

% Original transforms estimated on the 36x36x24 vNav grid
tform_matrix_vnav    = cell(1,n_vnav);
tform_matrixInv_vnav = cell(1,n_vnav);

% Reference = first vNav
img_N = abs(vnav(:,:,:,1));
img_N = img_N / prctile(abs(img_N(:)),75);

for n = 1:n_vnav

    img_R = abs(vnav(:,:,:,n));
    img_R = img_R / prctile(abs(img_R(:)),75);

    if n == 1

        tform_matrix_vnav{n} = imregtform( ...
            abs(img_N), ...
            abs(img_N), ...
            transformType, ...
            optimizer, ...
            metric);

    else

        % Same convention as previous QALAS motion pipeline:
        % reference pose -> current pose
        tform_matrix_vnav{n} = imregtform( ...
            abs(img_N), ...
            abs(img_R), ...
            transformType, ...
            optimizer, ...
            metric);

    end

    tform_matrixInv_vnav{n} = ...
        invert(tform_matrix_vnav{n});

    fprintf('vNav %2d / %2d registered\n',n,n_vnav);

end


%% Convert vNav-grid transforms to equivalent QALAS-grid transforms

fprintf('\nConverting vNav transforms to QALAS-grid transforms...\n');

voxel_nav_mm = 1000 * FOV ./ N_nav;
voxel_qalas_mm = 1000 * FOV ./ N_qalas;

fprintf('vNav voxel size:  %.4f x %.4f x %.4f mm\n', ...
    voxel_nav_mm(1),voxel_nav_mm(2),voxel_nav_mm(3));

fprintf('QALAS voxel size: %.4f x %.4f x %.4f mm\n', ...
    voxel_qalas_mm(1),voxel_qalas_mm(2),voxel_qalas_mm(3));


% MATLAB world axes:
% x_world = array dim 2
% y_world = array dim 1
% z_world = array dim 3

P = [0 1 0;
     1 0 0;
     0 0 1];

center_nav = [double(N_nav(1))/2 + 1;
              double(N_nav(2))/2 + 1;
              double(N_nav(3))/2 + 1];

center_qalas = [double(N_qalas(1))/2 + 1;
                double(N_qalas(2))/2 + 1;
                double(N_qalas(3))/2 + 1];

tform_matrix    = cell(1,n_vnav);
tform_matrixInv = cell(1,n_vnav);

translation_mm = zeros(n_vnav,3);
rotation_deg = zeros(n_vnav,3);

conversion_error_mm = zeros(n_vnav,1);

for n = 1:n_vnav

    tf_nav = tform_matrix_vnav{n};

    R_world_nav = double(tf_nav.R);
    t_world_nav = double(tf_nav.Translation(:));

    % MATLAB world coordinates -> array coordinates
    R_array = P.' * R_world_nav * P;
    t_array_nav = P.' * t_world_nav;

    % Center-referenced translation on vNav grid
    t_eff_nav = ...
        t_array_nav + ...
        (R_array - eye(3)) * center_nav;

    % Physical translation in mm
    t_eff_mm = ...
        t_eff_nav .* voxel_nav_mm(:);

    % Same physical translation in QALAS voxels
    t_eff_qalas = ...
        t_eff_mm ./ voxel_qalas_mm(:);

    % Raw affine translation required on QALAS grid
    t_array_qalas = ...
        t_eff_qalas - ...
        (R_array - eye(3)) * center_qalas;

    % Back to MATLAB world-axis convention
    R_world_qalas = ...
        P * R_array * P.';

    t_world_qalas = ...
        P * t_array_qalas;

    tform_matrix{n} = rigidtform3d( ...
        R_world_qalas, ...
        t_world_qalas.');

    tform_matrixInv{n} = ...
        invert(tform_matrix{n});


    % Verify conversion
    R_array_check = ...
        P.' * double(tform_matrix{n}.R) * P;

    t_array_check = ...
        P.' * double(tform_matrix{n}.Translation(:));

    t_eff_check = ...
        t_array_check + ...
        (R_array_check - eye(3)) * center_qalas;

    t_eff_check_mm = ...
        t_eff_check .* voxel_qalas_mm(:);

    conversion_error_mm(n) = ...
        norm(t_eff_check_mm - t_eff_mm);

    assert( ...
        norm(R_array_check - R_array,'fro') < 1e-10, ...
        'Rotation conversion failed for vNav %d.',n);

    translation_mm(n,:) = ...
        t_eff_mm.';


    % Rotation angles for plotting
    R = R_array;

    sy = -R(3,1);
    sy = max(-1,min(1,sy));

    ry = asin(sy);
    cy = cos(ry);

    if abs(cy) > 1e-8

        rx = atan2(R(3,2),R(3,3));
        rz = atan2(R(2,1),R(1,1));

    else

        rx = atan2(-R(2,3),R(2,2));
        rz = 0;

    end

    rotation_deg(n,:) = ...
        rad2deg([rx ry rz]);

end

fprintf('Maximum transform conversion error: %.6e mm\n', ...
    max(conversion_error_mm));

assert( ...
    max(conversion_error_mm) < 1e-6, ...
    'vNav -> QALAS transform conversion failed.');


%% Registration quality

fprintf('\nChecking registration quality...\n');

Rout = imref3d(size(img_N));

error_before = zeros(n_vnav,1);
error_after  = zeros(n_vnav,1);

for n = 1:n_vnav

    img_R = abs(vnav(:,:,:,n));
    img_R = img_R / prctile(abs(img_R(:)),75);

    error_before(n) = ...
        norm(img_R(:)-img_N(:)) / ...
        norm(img_N(:));

    img_R2N = imwarp( ...
        img_R, ...
        tform_matrixInv_vnav{n}, ...
        'cubic', ...
        'OutputView',Rout, ...
        'FillValues',0);

    error_after(n) = ...
        norm(img_R2N(:)-img_N(:)) / ...
        norm(img_N(:));

end


%% Plot physical motion

figure('Color','w');

subplot(2,1,1);

plot(1:n_vnav,rotation_deg,'LineWidth',1.5);

grid on;
xlim([1 n_vnav]);

xlabel('vNav number');
ylabel('Rotation (deg)');

legend( ...
    'R_x', ...
    'R_y', ...
    'R_z', ...
    'Location','best');

title('No-motion scan: vNav motion estimation');


subplot(2,1,2);

plot(1:n_vnav,translation_mm,'LineWidth',1.5);

grid on;
xlim([1 n_vnav]);

xlabel('vNav number');
ylabel('Translation (mm)');

legend( ...
    'T_x', ...
    'T_y', ...
    'T_z', ...
    'Location','best');

exportgraphics( ...
    gcf, ...
    fullfile(out_dir,'motion_trajectory_vnav.png'), ...
    'Resolution',200);


%% Print motion summary

translation_norm_mm = ...
    sqrt(sum(translation_mm.^2,2));

rotation_norm_deg = ...
    sqrt(sum(rotation_deg.^2,2));

fprintf('\nNo-motion scan vNav summary:\n');

fprintf('Maximum |Tx|: %.3f mm\n', ...
    max(abs(translation_mm(:,1))));

fprintf('Maximum |Ty|: %.3f mm\n', ...
    max(abs(translation_mm(:,2))));

fprintf('Maximum |Tz|: %.3f mm\n', ...
    max(abs(translation_mm(:,3))));

fprintf('Maximum translation magnitude: %.3f mm\n', ...
    max(translation_norm_mm));

fprintf('Maximum |Rx|: %.3f deg\n', ...
    max(abs(rotation_deg(:,1))));

fprintf('Maximum |Ry|: %.3f deg\n', ...
    max(abs(rotation_deg(:,2))));

fprintf('Maximum |Rz|: %.3f deg\n', ...
    max(abs(rotation_deg(:,3))));

fprintf('Maximum rotation magnitude: %.3f deg\n', ...
    max(rotation_norm_deg));


%% Plot registration quality

figure('Color','w');

plot( ...
    1:n_vnav, ...
    error_before, ...
    'LineWidth',1.5);

hold on;

plot( ...
    1:n_vnav, ...
    error_after, ...
    'LineWidth',1.5);

hold off;

grid on;
xlim([1 n_vnav]);

xlabel('vNav number');
ylabel('Relative image difference');

legend( ...
    'Before registration', ...
    'After registration', ...
    'Location','best');

title('No-motion scan: vNav registration quality');

exportgraphics( ...
    gcf, ...
    fullfile(out_dir,'registration_quality_vnav.png'), ...
    'Resolution',200);


%% Visual registration checks

check_navs = [20 40];

z = round(size(img_N,3)/2);

for kk = 1:numel(check_navs)

    n = check_navs(kk);

    img_R = abs(vnav(:,:,:,n));
    img_R = img_R / prctile(abs(img_R(:)),75);

    img_R2N = imwarp( ...
        img_R, ...
        tform_matrixInv_vnav{n}, ...
        'cubic', ...
        'OutputView',Rout, ...
        'FillValues',0);

    figure('Color','w');

    subplot(2,3,1);
    imagesc(img_N(:,:,z));
    axis image off;
    colormap gray;
    title('vNav 1');

    subplot(2,3,2);
    imagesc(img_R(:,:,z));
    axis image off;
    title(sprintf('vNav %d before',n));

    subplot(2,3,3);
    imagesc(img_R2N(:,:,z));
    axis image off;
    title(sprintf('vNav %d after',n));

    subplot(2,3,4);
    imagesc(abs(img_N(:,:,z)-img_R(:,:,z)));
    axis image off;
    title('Difference before');

    subplot(2,3,5);
    imagesc(abs(img_N(:,:,z)-img_R2N(:,:,z)));
    axis image off;
    title('Difference after');

    subplot(2,3,6);
    imagesc(img_N(:,:,z)-img_R2N(:,:,z));
    axis image off;
    title('Signed difference');

    sgtitle(sprintf( ...
        'No-motion scan: vNav %d registration to vNav 1',n));

end


%% Save comprehensive motion file

save( ...
    fullfile(out_dir,'motion_estimation.mat'), ...
    'tform_matrix', ...
    'tform_matrixInv', ...
    'tform_matrix_vnav', ...
    'tform_matrixInv_vnav', ...
    'rotation_deg', ...
    'translation_mm', ...
    'translation_norm_mm', ...
    'rotation_norm_deg', ...
    'conversion_error_mm', ...
    'error_before', ...
    'error_after', ...
    'vnav', ...
    'FOV', ...
    'N_nav', ...
    'N_qalas', ...
    'voxel_nav_mm', ...
    'voxel_qalas_mm', ...
    '-v7.3');


%% Save compatibility file

save( ...
    fullfile(out_dir,'_moco_1TR_motion.mat'), ...
    'tform_matrix', ...
    '-v7.3');

fprintf('\nMotion estimation saved.\n');

fprintf('Comprehensive file:\n%s\n', ...
    fullfile(out_dir,'motion_estimation.mat'));

fprintf('Downstream-compatible motion file:\n%s\n', ...
    fullfile(out_dir,'_moco_1TR_motion.mat'));


%% Functions

function ind = center_indices(N,n)

start_ind = floor((N-n)/2) + 1;

ind = start_ind:(start_ind+n-1);

end


function data_cc = apply_coil_compression(data,W)

sz = size(data);

x = permute(data,[1 3 2]);

x = reshape(x,[],sz(2));

x = x * W;

data_cc = reshape( ...
    x, ...
    sz(1), ...
    sz(3), ...
    size(W,2));

data_cc = permute(data_cc,[1 3 2]);

end


function data = readraw(file,pos,NCol,NCha,refl)

fid = fopen(file,'r');

if fid < 0
    error('Could not open raw data file.');
end

data = complex(zeros( ...
    NCol, ...
    NCha, ...
    numel(pos), ...
    'single'));

for i = 1:numel(pos)

    fseek(fid,double(pos(i))+192,'bof');

    x = fread( ...
        fid, ...
        [2 (NCol+4)*NCha], ...
        'float32=>single');

    x = complex(x(1,:),x(2,:));

    x = reshape(x,[NCol+4 NCha]);

    x = x(5:end,:);

    if refl(i)
        x = flip(x,1);
    end

    data(:,:,i) = x;

end

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

fwrite( ...
    fid, ...
    [real(x).'; imag(x).'], ...
    'float32');

fclose(fid);

end


function x = rcfl(name)

fid = fopen([name '.hdr']);

fgetl(fid);

d = sscanf(fgetl(fid),'%d').';

fclose(fid);

last = find(d~=1,1,'last');

if isempty(last)
    d = 1;
else
    d = d(1:last);
end

N = prod(d);

fid = fopen([name '.cfl']);

a = fread( ...
    fid, ...
    [2 N], ...
    'float32=>single');

fclose(fid);

x = reshape( ...
    complex(a(1,:),a(2,:)), ...
    d);

end