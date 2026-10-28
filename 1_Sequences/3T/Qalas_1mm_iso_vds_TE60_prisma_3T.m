% 3D Spiral-QALAS sequence generation for 3T Siemens Prisma
% Five QALAS contrasts with a Cartesian reference scan and 3D spiral vNavs.
% Requires Pulseq, the vds spiral designer, accompanying MATLAB helpers,
% and preparation-pulse waveforms in csv_imports/.

clear; close all; clc;

%% Configuration
save_seq    = 1;
see_seqeyes = 1;
cal_pns     = 1;

sys_type = 'prisma';
seq_name_to_save = 'qalas_spiral_1mm_vds_slab_sel_TGA_R1_optgap_v7_nav_strictFA_prisma';

% Dependencies
restoredefaultpath;
addpath('/autofs/cluster/berkin/yuting/MATLAB/pulseq_seq/spiral_traj/vdspiral/vdspiral')
addpath(genpath('/autofs/cluster/berkin/unay/mysoftware/pulseq_v151'));

% Hardware safety margins
slew_safety_margin          = 0.8;
grad_safety_margin          = 0.8;
spiral_slew_safety_margin   = 0.6;
spiral_grad_safety_margin   = 0.8;
lowPNS_slew_safety_margin   = 0.9;
lowPNS_grad_safety_margin   = 0.9;

MAX_ADCS = 100;

adcDwellSpiralDesign = 1e-6;

% Preserve the nominal dwell for independent navigator ADC design.
adcDwell_nominal = 1.2e-6;   % [s] - used for both QALAS and navigator ADC design
adcDwell         = adcDwell_nominal;

% Imaging geometry
fov             = [220 220 144]*1e-3;
N               = [220 220 144];
gradOversampling = false;
Nx = N(1); Ny = N(2); Nz = N(3);
deltak = 1./fov;

% Cartesian reference scan parameters
Ny_ref      = 32;
Nz_ref      = 32;
Ndummy_ref  = 50;
TE_ref      = 6e-3;
TR_ref      = 13e-3;
adcDwell_ref = 14e-6;
Tread_ref   = adcDwell_ref * Nx;

% QALAS acquisition parameters
slabThickness   = fov(3)*0.9;
N_interl        = 64;
R               = 1;
Ry              = 1;
Rz              = 1;

dwell           = 20e-6;
Tread           = dwell * Nz;
Tpre            = 3e-3;
flip_angle      = 4;
rot_angle       = 23.628/180*pi;
rfSpoilingInc   = 117;
TR_FLASH        = 7.07e-3;
TE              = [1.50]*1e-3;
gap_between_readouts = 1088e-3;
n_rep           = 1;
kz_co           = 1;
kzCAIPI         = 1;
kzseq           = 0;

% Spiral volumetric navigator parameters
num_NAV             = 1;        % 0=OFF, 1=ON
nav_res_nom         = 6e-3;     % nominal iso resolution [m]
fov_nav             = fov;
N_nav               = 2 * round(fov_nav ./ (2*nav_res_nom));
Nx_nav = N_nav(1); Ny_nav = N_nav(2); Nz_nav = N_nav(3);
nav_res_actual      = fov_nav ./ N_nav;
deltak_nav          = 1 ./ fov_nav;
nav_after_contrast  = [0 0 0 0 1];  % insert vNav after contrast 5 only
delay_before_vNav   = 10e-3;
delay_after_vNav    = 10e-3;

assert(nav_res_nom > 0, 'nav_res_nom must be positive.');
fprintf('\n3D spiral vNav geometry:\n');
fprintf('  FOV:        %.1f x %.1f x %.1f mm\n', fov_nav*1e3);
fprintf('  Matrix:     %d x %d x %d\n', N_nav);
fprintf('  Resolution: %.2f x %.2f x %.2f mm\n', nav_res_actual*1e3);

% Fat saturation and acquisition flags
num_FS      = 0;
Nz_shot     = 1;
num_part    = 1;

enable_FS  = (num_FS  > 0);
enable_NAV = (num_NAV > 0);

%% Scanner configuration and excitation
if strcmp(sys_type,'prisma') || strcmp(sys_type,'C2_simulate_prisma') || strcmp(sys_type,'prisma_XA30A')
    physical_slew_max = 200; physical_grad_max = 80; B0 = 2.89;
elseif strcmp(sys_type,'Connectome2')
    physical_slew_max = 598.802; physical_grad_max = 500; B0 = 2.89;
elseif strcmp(sys_type,'skyra')
    physical_slew_max = 180; physical_grad_max = 43; B0 = 2.89;
elseif strcmp(sys_type,'trio')
    physical_slew_max = 170; physical_grad_max = 38; B0 = 2.89;
elseif strcmp(sys_type,'vida')
    physical_slew_max = 200; physical_grad_max = 60; B0 = 2.89;
else
    error('Undefined sys_type');
end

slew_max = physical_slew_max * slew_safety_margin;
grad_max = physical_grad_max * grad_safety_margin;

sys = mr.opts('MaxGrad', grad_max, 'GradUnit', 'mT/m', ...
    'MaxSlew', slew_max, 'SlewUnit', 'T/m/s', ...
    'rfRingdownTime', 30e-6, 'rfDeadtime', 100e-6, 'adcDeadTime', 10e-6, 'B0', B0);

sys_lowPNS = mr.opts('MaxGrad', physical_grad_max*lowPNS_grad_safety_margin, 'GradUnit', 'mT/m', ...
    'MaxSlew', physical_slew_max*lowPNS_slew_safety_margin, 'SlewUnit', 'T/m/s', ...
    'rfRingdownTime', 30e-6, 'rfDeadtime', 100e-6, 'adcDeadTime', 10e-6, 'B0', B0);

sys_spi = mr.opts('MaxGrad', physical_grad_max*spiral_grad_safety_margin, 'GradUnit', 'mT/m', ...
    'MaxSlew', physical_slew_max*spiral_slew_safety_margin, 'SlewUnit', 'T/m/s', ...
    'rfRingdownTime', 40e-6, 'rfDeadtime', 100e-6, 'adcDeadTime', 10e-6, 'B0', B0);

seq = mr.Sequence(sys_spi);
sys = sys_spi;

% Slab-selective excitation
[rf, gz, gzReph] = mr.makeSincPulse( ...
    flip_angle*pi/180, 'system', sys, 'Duration', 1e-3, ...
    'SliceThickness', slabThickness, 'apodization', 0.5, ...
    'timeBwProduct', 16, 'use', 'excitation');

rf_ref = rf;

%% Cartesian reference scan design
gx_ref = mr.makeTrapezoid('x', sys, ...
    'Amplitude', Nx*deltak(1)/Tread_ref, ...
    'FlatTime', ceil(Tread_ref/sys.gradRasterTime)*sys.gradRasterTime);
gxPre_ref   = mr.makeTrapezoid('x', sys, 'Area', -gx_ref.area/2);
gxSpoil_ref = mr.makeTrapezoid('x', sys, 'Area',  gx_ref.area);
adc_ref     = mr.makeAdc(Nx, 'Duration', Tread_ref, 'Delay', gx_ref.riseTime, 'system', sys);

Texc_ref        = mr.calcDuration(rf_ref, gz);
tRFcenter_ref   = rf_ref.delay + mr.calcRfCenter(rf_ref);
tADCcenter_ref  = adc_ref.delay + Tread_ref/2;
TslabReph_ref   = mr.calcDuration(gzReph);
Tpre_ref        = mr.calcDuration(gxPre_ref);
TreadBlock_ref  = mr.calcDuration(gx_ref, adc_ref);
Tpost_ref       = mr.calcDuration(gxSpoil_ref);

delayTE_ref = ceil( ...
    (TE_ref - (Texc_ref - tRFcenter_ref) - TslabReph_ref - Tpre_ref - tADCcenter_ref) ...
    / seq.gradRasterTime) * seq.gradRasterTime;
assert(delayTE_ref >= 0, 'Reference-scan TE too short.');

delayTR_ref = ceil( ...
    (TR_ref - Texc_ref - TslabReph_ref - Tpre_ref - delayTE_ref - TreadBlock_ref - Tpost_ref) ...
    / seq.gradRasterTime) * seq.gradRasterTime;
assert(delayTR_ref >= 0, 'Reference-scan TR too short.');

fprintf('\nReference scan TE delay: %.3f ms\n', delayTE_ref*1e3);
fprintf('Reference scan TR delay: %.3f ms\n', delayTR_ref*1e3);

areaY_ref = ((0:Ny-1) - Ny/2) * deltak(2);
areaZ_ref = ((0:Nz-1) - Nz/2) * deltak(3);

tmpY_ref = 1:Ny;
iY_ref_indices = tmpY_ref(1 + end/2 - Ny_ref/2 : end/2 + Ny_ref/2);
tmpZ_ref = 1:Nz;
iZ_ref_indices = tmpZ_ref(1 + end/2 - Nz_ref/2 : end/2 + Nz_ref/2);

for iY = iY_ref_indices
    gpeYPre_ref(iY)  = mr.makeTrapezoid('y', 'Area',  areaY_ref(iY), ...
        'Duration', mr.calcDuration(gxPre_ref),   'system', sys);
    gpeYReph_ref(iY) = mr.makeTrapezoid('y', 'Area', -areaY_ref(iY), ...
        'Duration', mr.calcDuration(gxSpoil_ref), 'system', sys);
end

% Fat-saturation module
sat_ppm     = -3.45;
sat_freq    = sat_ppm*1e-6*B0*sys.gamma;
rf_fs       = mr.makeGaussPulse(110*pi/180, 'system', sys, 'Duration', 8e-3, ...
    'bandwidth', abs(sat_freq), 'freqOffset', sat_freq);
rf_fs.phaseOffset = -2*pi*rf_fs.freqOffset*mr.calcRfCenter(rf_fs);

spoiler_amp = 3*8*42.58*10e2;
est_rise    = 500e-6;
est_flat    = 2500e-6;

gp_r = mr.makeTrapezoid('x','amplitude', spoiler_amp,'riseTime',est_rise,'flatTime',est_flat,'system',sys_lowPNS);
gp_p = mr.makeTrapezoid('y','amplitude', spoiler_amp,'riseTime',est_rise,'flatTime',est_flat,'system',sys_lowPNS);
gp_s = mr.makeTrapezoid('z','amplitude', spoiler_amp,'riseTime',est_rise,'flatTime',est_flat,'system',sys_lowPNS);
gn_r = mr.makeTrapezoid('x','amplitude',-spoiler_amp,'delay',mr.calcDuration(rf_fs),'riseTime',est_rise,'flatTime',est_flat,'system',sys_lowPNS);
gn_p = mr.makeTrapezoid('y','amplitude',-spoiler_amp,'delay',mr.calcDuration(rf_fs),'riseTime',est_rise,'flatTime',est_flat,'system',sys_lowPNS);
gn_s = mr.makeTrapezoid('z','amplitude',-spoiler_amp,'delay',mr.calcDuration(rf_fs),'riseTime',est_rise,'flatTime',est_flat,'system',sys_lowPNS);

% kz-encoding gradients
if kzCAIPI == 1
    for itl_id = 1:N_interl*n_rep/R
        [gz_blips, slice_indices] = prepare_gz_blips_group_CAIPI(Nz,Rz,Nz_shot,kz_co,deltak,sys,itl_id);
        gz_blips_itl(itl_id,:)       = gz_blips;
        slice_indices_itl(itl_id,:)  = slice_indices;
        slice_indices_icd_mid        = slice_indices_itl;
    end
else
    [gz_blips, slice_indices] = prepare_gz_blips_group(Nz,Rz,Nz_shot,kz_co,deltak,sys);
    gz_blips_itl              = gz_blips;
    slice_indices_icd_mid     = slice_indices;
end

no_blip_delay = mr.makeDelay(mr.calcDuration(gz_blips(3)));

app = zeros(Nz,1);
for ii = 1:Nz
    if isempty(gz_blips(ii).amplitude)
        app(ii) = 0;
    else
        app(ii) = gz_blips(ii).amplitude;
    end
end
figure; plot(app); hold on; plot(sort(app)); title('gz blip amplitudes');

%% QALAS spiral trajectory and ADC
res  = 1e2 * fov(1) / Nx;
rmax = 1 / (2*res);

F0 = fov(1)*100; m = 2.0; uc = 0.15; p = 3;
u    = linspace(0,1,2000).';
Ftar = zeros(size(u));
idx  = (u <= uc);
Ftar(idx)  = m*F0;
uu         = u(~idx);
Ftar(~idx) = F0*(1 + (m-1)*((1-uu)/(1-uc)).^p);

M = 5;
A = zeros(length(u), M+1);
for k = 0:M; A(:,k+1) = u.^k; end

Aeq = zeros(3,M+1); beq = zeros(3,1);
Aeq(1,1) = 1; beq(1) = m*F0;
Aeq(2,:) = 1; beq(2) = F0;
if M >= 1; Aeq(3,2) = 1; end

Du    = diff(A,1,1);
Apos  = -A; bpos = -F0*ones(size(u));
Aineq = [Du; Apos]; bineq = [zeros(size(Du,1),1); bpos];

opts   = optimoptions('lsqlin','Display','off');
c      = lsqlin(A, Ftar, Aineq, bineq, Aeq, beq, [], [], [], opts);
Fcoeff = c.';

disp('Calculating QALAS spiral gradient...');
[k,g,s,~,~,~] = vds(0.59*physical_slew_max*100, spiral_grad_safety_margin*physical_grad_max/10, ...
    adcDwellSpiralDesign, N_interl, Fcoeff, rmax);

uSF = round(sys.gradRasterTime/adcDwellSpiralDesign);
k = k(1:uSF:end); g = g(1:uSF:end);
g = [real(g(:)) imag(g(:))];
g_mT_m   = 10*g;
k_std    = k*100;
grad_std = mr.convert(g_mT_m,'mT/m');

spiral_grad_shape = grad_std.';

max_allowed_grad = 1/adcDwell_nominal/max(fov)/sys.gamma;
grad_std_sos = sqrt(grad_std(:,1).^2 + grad_std(:,2).^2);
grad_sos_mTm = mr.convert(max(grad_std_sos),'Hz/m','mT/m');
assert(grad_sos_mTm <= max_allowed_grad*1e3, ...
    'QALAS readout gradient too large (angular undersampling).');

% ADC segmentation
% The QALAS ADC dwell is adjusted below while adcDwell_nominal is retained.
adcTime = sys.gradRasterTime * size(spiral_grad_shape,2);
adcSamplesDesired = round(adcTime / adcDwell);

adcSamplesPerSegment = ceil(adcSamplesDesired / MAX_ADCS);
adcSamplesPerSegment = ceil(adcSamplesPerSegment / 100) * 100;
if adcSamplesDesired / adcSamplesPerSegment > MAX_ADCS
    error('QALAS ADC exceeds MAX_ADCS (%d)', MAX_ADCS);
end

adcSegments = fix(adcSamplesDesired / adcSamplesPerSegment);
adcSamples  = adcSegments * adcSamplesPerSegment;
adcDwell    = round(adcTime/adcSamples/100e-9)*100e-9;  % overwrites adcDwell_nominal scope
if adcDwell < 1e-6; error('QALAS ADC dwell < 1 us'); end

adcSegmentDuration = adcSamplesPerSegment * adcDwell;
if abs(adcSegmentDuration/sys.gradRasterTime - round(adcSegmentDuration/sys.gradRasterTime)) > 1e-10
    error('QALAS ADC segment duration not aligned to gradRasterTime');
end
adcSegments = floor(adcTime / adcSegmentDuration);
adcSamples  = adcSegments * adcSamplesPerSegment;

add_ADC_sample            = 10;
add_zero_time             = add_ADC_sample * adcDwell;
adcSamples                = adcSamples + add_ADC_sample;
spiral_grad_shape_add_pot = round(add_zero_time / sys.gradRasterTime);
spiral_grad_add           = zeros(2, spiral_grad_shape_add_pot);

adc = mr.makeAdc(adcSamples, 'Dwell', adcDwell, 'Delay', sys.adcDeadTime);
fprintf('\nQALAS ADC: %d samples, dwell=%.3f us\n', adcSamples, adcDwell*1e6);

if ~gradOversampling
    spiral_grad_shape = [spiral_grad_add, spiral_grad_shape, spiral_grad_shape(:,end)];
else
    spiral_grad_shape = [spiral_grad_shape, spiral_grad_shape(:,end), spiral_grad_shape(:,end)];
    if mod(length(spiral_grad_shape),2)==0
        spiral_grad_shape = [spiral_grad_shape, spiral_grad_shape(:,end)];
    end
    spiral_grad_shape = [spiral_grad_add, spiral_grad_shape];
end

gx_spi_out = mr.makeArbitraryGrad('x', spiral_grad_shape(1,:), ...
    'Delay', sys.adcDeadTime, 'first', 0, 'last', spiral_grad_shape(1,end), 'system', sys);
gy_spi_out = mr.makeArbitraryGrad('y', spiral_grad_shape(2,:), ...
    'Delay', sys.adcDeadTime, 'first', 0, 'last', spiral_grad_shape(2,end), 'system', sys);

fprintf('<strong>QALAS spiral duration: %.1f ms</strong>\n', mr.calcDuration(gx_spi_out,gy_spi_out)*1e3);

gx_spi_in = mr.makeArbitraryGrad('x', -spiral_grad_shape(1,end:-1:1), sys);
gy_spi_in = mr.makeArbitraryGrad('y', -spiral_grad_shape(2,end:-1:1), sys);

gx_rwd_dur = ceil(abs(spiral_grad_shape(1,end))/sys.maxSlew/seq.gradRasterTime)*seq.gradRasterTime;
gy_rwd_dur = ceil(abs(spiral_grad_shape(2,end))/sys.maxSlew/seq.gradRasterTime)*seq.gradRasterTime;
rwd_dur    = max(gx_rwd_dur, gy_rwd_dur);

gx_rewinder_out = mr.makeExtendedTrapezoid('x','times',[0 rwd_dur],'amplitudes',[spiral_grad_shape(1,end) 0]);
gy_rewinder_out = mr.makeExtendedTrapezoid('y','times',[0 rwd_dur],'amplitudes',[spiral_grad_shape(2,end) 0]);
gx_rewinder_in  = mr.makeExtendedTrapezoid('x','times',[0 rwd_dur],'amplitudes',[0 spiral_grad_shape(1,end)]);
gy_rewinder_in  = mr.makeExtendedTrapezoid('y','times',[0 rwd_dur],'amplitudes',[0 spiral_grad_shape(2,end)]);

if length(TE) > 1
    try
        gx_to_ksp_center = mr.makeExtendedTrapezoidArea('x', gx_spi_out.waveform(end), 0, -gx_spi_out.area, sys_lowPNS);
        gy_to_ksp_center = mr.makeExtendedTrapezoidArea('y', gy_spi_out.waveform(end), 0, -gy_spi_out.area, sys_lowPNS);
    catch
        tic;
        [gx_to_ksp_center,gy_to_ksp_center,tmps,tmpg,tmpt] = grid_search_rewinder( ...
            physical_grad_max*lowPNS_grad_safety_margin, physical_slew_max*lowPNS_slew_safety_margin, ...
            gx_spi_out.waveform(end), gy_spi_out.waveform(end), -gx_spi_out.area, -gy_spi_out.area, sys);
        fprintf('Grid search rewinder: %.1fs, dur=%.1fms\n', toc, tmpt*1e3);
    end
    gx_rewinder_out = gx_to_ksp_center;
    gy_rewinder_out = gy_to_ksp_center;
end

gz_spoil = mr.makeTrapezoid('z', sys, 'Area', deltak(1)*Nx*4);
gx_spoil = mr.makeExtendedTrapezoid('x','times',[0 mr.calcDuration(gz_spoil)],'amplitudes',[spiral_grad_shape(1,end) 0],'system',sys);
gy_spoil = mr.makeExtendedTrapezoid('y','times',[0 mr.calcDuration(gz_spoil)],'amplitudes',[spiral_grad_shape(2,end) 0],'system',sys);

%% Spiral navigator trajectory and ADC
N_interl_nav    = 1;
res_nav_cm      = 1e2 * fov_nav(1) / Nx_nav;
rmax_nav        = 1 / (2 * res_nav_cm);
F0_nav          = fov_nav(1) * 100;
Fcoeff_nav      = [F0_nav 0 0 0 0 0];

fprintf('\nDesigning navigator spiral...\n');
[k_nav, g_nav, ~, ~, ~, ~] = vds( ...
    0.59*physical_slew_max*100, spiral_grad_safety_margin*physical_grad_max/10, ...
    adcDwellSpiralDesign, N_interl_nav, Fcoeff_nav, rmax_nav);

uSF_nav = round(sys.gradRasterTime / adcDwellSpiralDesign);
k_nav = k_nav(1:uSF_nav:end);
g_nav = g_nav(1:uSF_nav:end);
g_nav = [real(g_nav(:)) imag(g_nav(:))];

g_nav_mT_m       = 10*g_nav;
grad_nav_std     = mr.convert(g_nav_mT_m,'mT/m');
spiral_grad_shape_nav = grad_nav_std.';

% Navigator ADC segmentation
%
% Use the original nominal dwell for navigator ADC design.
adcTime_nav = sys.gradRasterTime * size(spiral_grad_shape_nav,2);

adcSamplesDesired_nav = round(adcTime_nav / adcDwell_nominal);

adcSamplesPerSegment_nav = ceil(adcSamplesDesired_nav / MAX_ADCS);
adcSamplesPerSegment_nav = ceil(adcSamplesPerSegment_nav / 100) * 100;

if adcSamplesDesired_nav / adcSamplesPerSegment_nav > MAX_ADCS
    error('Navigator ADC exceeds MAX_ADCS (%d). Reduce nav resolution or increase adcDwell_nominal.', MAX_ADCS);
end

adcSegments_nav = fix(adcSamplesDesired_nav / adcSamplesPerSegment_nav);
adcSamples_nav  = adcSegments_nav * adcSamplesPerSegment_nav;
adcDwell_nav    = round(adcTime_nav / adcSamples_nav / 100e-9) * 100e-9;

if adcDwell_nav < 1e-6
    error('Navigator ADC dwell < 1 us. Likely to fail on scanner.');
end

adcSegmentDuration_nav = adcSamplesPerSegment_nav * adcDwell_nav;
if abs(adcSegmentDuration_nav/sys.gradRasterTime - round(adcSegmentDuration_nav/sys.gradRasterTime)) > 1e-10
    error('Navigator ADC segment duration not aligned to gradRasterTime.');
end

adcSegments_nav = floor(adcTime_nav / adcSegmentDuration_nav);
adcSamples_nav  = adcSegments_nav * adcSamplesPerSegment_nav;

add_ADC_sample_nav            = 10;
add_zero_time_nav             = add_ADC_sample_nav * adcDwell_nav;
adcSamples_nav                = adcSamples_nav + add_ADC_sample_nav;
spiral_grad_shape_add_pot_nav = round(add_zero_time_nav / sys.gradRasterTime);
spiral_grad_add_nav           = zeros(2, spiral_grad_shape_add_pot_nav);

% Prepend zeros, append repeated last sample
spiral_grad_shape_nav = [spiral_grad_add_nav, spiral_grad_shape_nav, spiral_grad_shape_nav(:,end)];

fprintf('Navigator ADC: %d samples (%d segs x %d), dwell=%.3f us\n', ...
    adcSamples_nav, adcSegments_nav, adcSamplesPerSegment_nav, adcDwell_nav*1e6);

% Store as a template; phaseOffset is applied per-shot inside addSpiralVNav
adc_nav_template = mr.makeAdc(adcSamples_nav, ...
    'Dwell', adcDwell_nav, 'Delay', sys.adcDeadTime, 'system', sys);

% Gradient and ADC timing alignment
gx_nav_tmp = mr.makeArbitraryGrad('x', spiral_grad_shape_nav(1,:), ...
    'Delay', sys.adcDeadTime, 'first', 0, 'last', spiral_grad_shape_nav(1,end), 'system', sys);
gy_nav_tmp = mr.makeArbitraryGrad('y', spiral_grad_shape_nav(2,:), ...
    'Delay', sys.adcDeadTime, 'first', 0, 'last', spiral_grad_shape_nav(2,end), 'system', sys);

T_grad_nav_event = mr.calcDuration(gx_nav_tmp, gy_nav_tmp);
T_adc_nav_event  = mr.calcDuration(adc_nav_template);

if T_grad_nav_event < T_adc_nav_event
    n_extra_nav = ceil((T_adc_nav_event - T_grad_nav_event) / sys.gradRasterTime);
    spiral_grad_shape_nav = [spiral_grad_shape_nav, repmat(spiral_grad_shape_nav(:,end),1,n_extra_nav)];
    fprintf('Extended navigator gradient by %d sample(s).\n', n_extra_nav);
end

gx_nav = mr.makeArbitraryGrad('x', spiral_grad_shape_nav(1,:), ...
    'Delay', sys.adcDeadTime, 'first', 0, 'last', spiral_grad_shape_nav(1,end), 'system', sys);
gy_nav = mr.makeArbitraryGrad('y', spiral_grad_shape_nav(2,:), ...
    'Delay', sys.adcDeadTime, 'first', 0, 'last', spiral_grad_shape_nav(2,end), 'system', sys);

fprintf('Navigator readout duration: %.3f ms\n', mr.calcDuration(gx_nav,gy_nav,adc_nav_template)*1e3);

% Navigator kz encoding
assert(mod(Nz_nav,2) == 0, 'Nz_nav must be even.');
kz_indices_nav  = (-Nz_nav/2):(Nz_nav/2-1);
areaZ_nav       = kz_indices_nav * deltak_nav(3);

gzEnc_nav_max   = mr.makeTrapezoid('z','Area', max(abs(areaZ_nav)), 'system', sys);
T_kzEnc_nav     = mr.calcDuration(gzEnc_nav_max);

gzEnc_nav = cell(1,Nz_nav);
for iZ_nav = 1:Nz_nav
    if abs(areaZ_nav(iZ_nav)) < eps
        gzEnc_nav{iZ_nav} = mr.makeDelay(T_kzEnc_nav);
    else
        gzEnc_nav{iZ_nav} = mr.makeTrapezoid('z', ...
            'Area', areaZ_nav(iZ_nav), 'Duration', T_kzEnc_nav, 'system', sys);
    end
end

fprintf('Navigator kz: %d partitions, enc duration %.3f ms\n', Nz_nav, T_kzEnc_nav*1e3);

% Navigator spoiling
nav_spoil_area  = 4 * Nz_nav * deltak_nav(3);
gzSpoil_nav_min = mr.makeTrapezoid('z','Area', nav_spoil_area, 'system', sys);
T_gzSpoil_nav   = mr.calcDuration(gzSpoil_nav_min);

T_xyRamp_nav = max( ...
    abs(spiral_grad_shape_nav(1,end)) / sys.maxSlew, ...
    abs(spiral_grad_shape_nav(2,end)) / sys.maxSlew);

T_spoil_nav = ceil(max(T_gzSpoil_nav, T_xyRamp_nav) / seq.gradRasterTime) * seq.gradRasterTime;

gzSpoil_nav = mr.makeTrapezoid('z','Area', nav_spoil_area, 'Duration', T_spoil_nav, 'system', sys);
gxSpoil_nav = mr.makeExtendedTrapezoid('x', ...
    'times', [0 T_spoil_nav], 'amplitudes', [spiral_grad_shape_nav(1,end) 0], 'system', sys);
gySpoil_nav = mr.makeExtendedTrapezoid('y', ...
    'times', [0 T_spoil_nav], 'amplitudes', [spiral_grad_shape_nav(2,end) 0], 'system', sys);

% Navigator shot timing
T_nav_shot_min = ...
    mr.calcDuration(rf,gz) + mr.calcDuration(gzReph) + T_kzEnc_nav + ...
    mr.calcDuration(gx_nav,gy_nav,adc_nav_template) + T_spoil_nav;

TR_nav_shot = ceil(T_nav_shot_min / seq.gradRasterTime) * seq.gradRasterTime;
delayTR_nav = TR_nav_shot - T_nav_shot_min;

assert(delayTR_nav >= 0, 'Navigator shot TR is negative - timing error.');

T_nav = Nz_nav * TR_nav_shot;

tRFcenter_nav = rf.delay + mr.calcRfCenter(rf);
TE_nav = (mr.calcDuration(rf,gz) - tRFcenter_nav) + mr.calcDuration(gzReph) + ...
         T_kzEnc_nav + adc_nav_template.delay;

fprintf('\nNavigator shot timing:\n');
fprintf('  TE: %.3f ms, shot TR: %.3f ms, total vol: %.3f ms\n', ...
    TE_nav*1e3, TR_nav_shot*1e3, T_nav*1e3);

%% Preparation pulses and QALAS timing
gslSp_t2prep = mr.makeTrapezoid( ...
    'z', ...
    'Amplitude', -42.58*8*1e3, ...
    'Risetime', 0.84e-3, ...
    'Duration', 0.84e-3+8e-3+0.84e-3, ...
    'system', sys);

gslSp_IRprep = mr.makeTrapezoid( ...
    'z', ...
    'Amplitude', -42.58*8*1e3, ...
    'Risetime', 1e-3, ...
    'Duration', 1e-3+8e-3+1e-3, ...
    'system', sys);

% Preparation RF pulses

% T2 preparation pulse
t2prep = readmatrix('csv_imports/T2prep.txt');
Re = t2prep(:,1) .* cos(t2prep(:,2));
Im = t2prep(:,1) .* sin(t2prep(:,2));

% Lengthened T2-preparation RF waveform to reduce peak B1
% while maintaining the prescribed flip angle.
t2prep_pulse = mr.makeArbitraryRf( ...
    (Re+Im*1i).', ...
    380.4*pi/180, ...
    'system', sys, ...
    'dwell', 2e-6);

% 90-degree preparation pulse
text = readmatrix('csv_imports/rf90.txt');
Re = text(:,1) .* cos(text(:,2));
Im = text(:,1) .* sin(text(:,2));

rf90 = mr.makeArbitraryRf( ...
    (Re+Im*1i).', ...
    pi/2, ...
    'system', sys, ...
    'dwell', 1e-6);

rf90_180PhaseOffset = mr.makeArbitraryRf( ...
    (Re+Im*1i).', ...
    pi/2, ...
    'system', sys, ...
    'dwell', 1e-6, ...
    'PhaseOffset', -180*pi/180);

% Inversion preparation pulse
IRprep = readmatrix('csv_imports/invpulse.txt');
Re = IRprep(:,1) .* cos(IRprep(:,2));
Im = IRprep(:,1) .* sin(IRprep(:,2));

% Lengthened inversion RF waveform to reduce peak B1
% while maintaining the prescribed flip angle.
IRprep_pulse = mr.makeArbitraryRf( ...
    (Re+Im*1i).', ...
    1500*pi/180, ...
    'system', sys, ...
    'dwell', 3e-5);

% Peak B1 diagnostics

fprintf('\nPeak B1 values:\n');
fprintf('Readout RF : %.2f uT\n', max(abs(rf.signal))/sys.gamma*1e6);
fprintf('RF90       : %.2f uT\n', max(abs(rf90.signal))/sys.gamma*1e6);
fprintf('T2prep     : %.2f uT\n', max(abs(t2prep_pulse.signal))/sys.gamma*1e6);
fprintf('IRprep     : %.2f uT\n', max(abs(IRprep_pulse.signal))/sys.gamma*1e6);

% QALAS timing
TE_min         = mr.calcDuration(rf)/2 + mr.calcDuration(gzReph) + mr.calcDuration(gz_blips(3));
delay_TE_dixon = TE - TE_min;

delayTR = ceil( ...
    (TR_FLASH - mr.calcDuration(rf) - mr.calcDuration(gzReph) - mr.calcDuration(gz_blips(3)) ...
     - mr.calcDuration(gx_spi_out) - mr.calcDuration(gz_spoil)) / seq.gradRasterTime) * seq.gradRasterTime;
fprintf('\ndelay TR: %.3f ms\n', delayTR*1e3);
assert(delayTR >= 0);

delayTR_dixon = delayTR - delay_TE_dixon;
assert(delayTR_dixon(end) >= 0);

T_fs  = mr.calcDuration(gp_r) + mr.calcDuration(gn_r);
nETL  = N(3)*num_part/Nz_shot*length(TE);
esp   = TR_FLASH;

% Set delays using the RF event durations to preserve pulse-center timing.

T_rf90   = mr.calcDuration(rf90);
T_t2prep = mr.calcDuration(t2prep_pulse);

blockRaster = sys.blockDurationRaster;

delay_1_t2prep = round( ...
    (11e-3 + T_rf90/2 - T_t2prep/2) ...
    / blockRaster) * blockRaster;

delay_2_t2prep = round( ...
    (25e-3 - T_t2prep) ...
    / blockRaster) * blockRaster;

delay_3_t2prep = round( ...
    (14e-3 - T_rf90/2 - T_t2prep/2) ...
    / blockRaster) * blockRaster;

assert(delay_1_t2prep >= 0);
assert(delay_2_t2prep >= 0);
assert(delay_3_t2prep >= 0);

delay_IRprep   = 100e-3 - mr.calcDuration(IRprep_pulse)/2 - T_fs*enable_FS;
delay_TRinner  = delayTR_dixon;

delay_TRouter = gap_between_readouts - esp*nETL - T_fs*num_FS;
delT_M3_M4    = gap_between_readouts - esp*nETL - mr.calcDuration(IRprep_pulse) ...
                - delay_IRprep - 0.22e-3 - T_fs*num_FS;
if delT_M3_M4 <= 1e-3; delT_M3_M4 = 1e-3; end

delT_M13_2end = gap_between_readouts - esp*nETL - 69.7e-3 - T_fs*num_FS;

assert(delay_TRouter  >= 0, 'delay_TRouter negative');
assert(delT_M13_2end  >= 0, 'delT_M13_2end negative');

T_nav_extra  = enable_NAV * (delay_before_vNav + T_nav + delay_after_vNav);
T_QALAS_cycle = numel(nav_after_contrast) * gap_between_readouts + T_nav_extra;

fprintf('QALAS cycle: %.3f s (%.3f min)\n', T_QALAS_cycle, T_QALAS_cycle/60);
fprintf('vNav extra time: %.3f ms\n', T_nav_extra*1e3);

% Group kz gradients into shots
if kzCAIPI
    gz_blips_group = reshape(gz_blips_itl, [N_interl/R*n_rep, Nz/Nz_shot, Nz_shot]);
else
    gz_blips_group = reshape(gz_blips_itl, [Nz/Nz_shot, Nz_shot]);
end

%% Sequence assembly
fprintf('\nAdding reference scan dummy TRs...\n');
[~, iY_ref_dummy] = min(abs(areaY_ref));
gpeYPre_ref_dummy  = mr.makeTrapezoid('y','Area', areaY_ref(iY_ref_dummy), ...
    'Duration',mr.calcDuration(gxPre_ref),'system',sys);
gpeYReph_ref_dummy = mr.makeTrapezoid('y','Area',-areaY_ref(iY_ref_dummy), ...
    'Duration',mr.calcDuration(gxSpoil_ref),'system',sys);

rf_phase_ref = 0; rf_inc_ref = 0;
for iDummy_ref = 1:Ndummy_ref
    rf_ref.phaseOffset = rf_phase_ref*pi/180;
    rf_inc_ref   = mod(rf_inc_ref   + rfSpoilingInc, 360);
    rf_phase_ref = mod(rf_phase_ref + rf_inc_ref,     360);
    seq.addBlock(rf_ref, gz);
    seq.addBlock(gzReph);
    seq.addBlock(gxPre_ref, gpeYPre_ref_dummy);
    seq.addBlock(mr.makeDelay(delayTE_ref));
    seq.addBlock(gx_ref);
    seq.addBlock(gpeYReph_ref_dummy, gxSpoil_ref);
    seq.addBlock(mr.makeDelay(delayTR_ref));
end

% Cartesian reference scan acquisition
fprintf('Adding Cartesian reference scan acquisition...\n');
ref_counter = 0;
for iZ = iZ_ref_indices
    gpeZPre_ref  = mr.makeTrapezoid('z','Area', areaZ_ref(iZ), ...
        'Duration',mr.calcDuration(gxPre_ref),'system',sys);
    gpeZReph_ref = mr.makeTrapezoid('z','Area',-areaZ_ref(iZ), ...
        'Duration',mr.calcDuration(gxSpoil_ref),'system',sys);
    for iY = iY_ref_indices
        ref_counter = ref_counter + 1;
        rf_ref.phaseOffset  = rf_phase_ref*pi/180;
        adc_ref.phaseOffset = rf_ref.phaseOffset;
        rf_inc_ref   = mod(rf_inc_ref   + rfSpoilingInc, 360);
        rf_phase_ref = mod(rf_phase_ref + rf_inc_ref,     360);
        seq.addBlock(rf_ref, gz);
        seq.addBlock(gzReph);
        seq.addBlock(gxPre_ref, gpeYPre_ref(iY), gpeZPre_ref);
        seq.addBlock(mr.makeDelay(delayTE_ref));
        seq.addBlock(gx_ref, adc_ref);
        seq.addBlock(gpeYReph_ref(iY), gpeZReph_ref, gxSpoil_ref);
        seq.addBlock(mr.makeDelay(delayTR_ref));
    end
end
fprintf('Reference readouts: %d / %d\n', ref_counter, Ny_ref*Nz_ref);

% Complementary spiral-arm ordering across contrasts
isDummy  = 1;
N_interl = round(N_interl/R);
fprintf('\nAcc: %d, interleaves: %d\n', R, N_interl);

ord_arm_c1 = permute(reshape(linspace(1,N_interl*n_rep,N_interl*n_rep),1,num_part,[]),[1,3,2]);
ord_arm_c2 = permute(reshape(linspace(N_interl*n_rep,1,N_interl*n_rep),1,num_part,[]),[1,3,2]);
ord_arm_c3 = permute(reshape(linspace(1,N_interl*n_rep,N_interl*n_rep),1,num_part,[]),[1,3,2]);
ord_arm_c4 = permute(reshape(linspace(N_interl*n_rep,1,N_interl*n_rep),1,num_part,[]),[1,3,2]);
ord_arm_c5 = permute(reshape(linspace(1,N_interl*n_rep,N_interl*n_rep),1,num_part,[]),[1,3,2]);
ord_arm    = cat(1,ord_arm_c1,ord_arm_c2,ord_arm_c3,ord_arm_c4,ord_arm_c5);
ord_group  = 1:Nz_shot;

% Dummy QALAS cycle
for ind_arm = 1
    for g = 1
        rf_phase = 0; rf_inc = 0;
        if kzCAIPI
            gz_blips_group_rd = reshape(gz_blips_group(ind_arm,:,:), Nz/Nz_shot, Nz_shot);
        else
            gz_blips_group_rd = gz_blips_group;
        end

        seq.addBlock(rf90, mr.makeDelay(delay_1_t2prep));
        seq.addBlock(t2prep_pulse, mr.makeDelay(delay_2_t2prep));
        seq.addBlock(t2prep_pulse, mr.makeDelay(delay_2_t2prep));
        seq.addBlock(t2prep_pulse, mr.makeDelay(delay_2_t2prep));
        seq.addBlock(t2prep_pulse, mr.makeDelay(delay_3_t2prep));
        seq.addBlock(rf90_180PhaseOffset);
        seq.addBlock(gslSp_t2prep);

        phi = (ord_arm(1,ind_arm)-1)*rot_angle;
        if enable_FS; seq.addBlock(gp_r,gp_p,gp_s); seq.addBlock(rf_fs,gn_r,gn_p,gn_s); end
        [rf_phase,rf_inc] = addAcq_FS(seq, nETL, rf, gz, gzReph, adc, rfSpoilingInc, ...
            rf_phase, rf_inc, gz_blips_group_rd, no_blip_delay, gx_spi_out, gy_spi_out, ...
            phi, delay_TE_dixon, delay_TRinner, gx_spoil, gy_spoil, gz_spoil, ...
            ord_group(g), isDummy, gp_r, gp_p, gp_s, rf_fs, gn_r, gn_p, gn_s, num_part, enable_FS);
        seq.addBlock(mr.makeDelay(delT_M3_M4));

        seq.addBlock(IRprep_pulse);
        seq.addBlock(gslSp_IRprep, mr.makeDelay(delay_IRprep));

        for ind_acq = 2:5
            phi = (ord_arm(ind_acq,ind_arm)-1)*rot_angle;
            if enable_FS; seq.addBlock(gp_r,gp_p,gp_s); seq.addBlock(rf_fs,gn_r,gn_p,gn_s); end
            [rf_phase,rf_inc] = addAcq_FS(seq, nETL, rf, gz, gzReph, adc, rfSpoilingInc, ...
                rf_phase, rf_inc, gz_blips_group_rd, no_blip_delay, gx_spi_out, gy_spi_out, ...
                phi, delay_TE_dixon, delay_TRinner, gx_spoil, gy_spoil, gz_spoil, ...
                ord_group(g), isDummy, gp_r, gp_p, gp_s, rf_fs, gn_r, gn_p, gn_s, num_part, enable_FS);
            if ind_acq < 5
                seq.addBlock(mr.makeDelay(delay_TRouter));
            else
                % contrast 5 end
                if delT_M13_2end > 0; seq.addBlock(mr.makeDelay(delT_M13_2end)); end
                if enable_NAV && nav_after_contrast(5)
                    seq.addBlock(mr.makeDelay(delay_before_vNav));
                    [rf_phase,rf_inc] = addSpiralVNav(seq, rf, gz, gzReph, ...
                        adc_nav_template, rfSpoilingInc, rf_phase, rf_inc, ...
                        gzEnc_nav, gx_nav, gy_nav, ...
                        gxSpoil_nav, gySpoil_nav, gzSpoil_nav, delayTR_nav, isDummy);
                    seq.addBlock(mr.makeDelay(delay_after_vNav));
                end
            end
        end
    end
end

isDummy = 0;

% Main QALAS acquisition
for ind_arm = 1:N_interl*n_rep
    for g = 1:Nz_shot
        rf_phase = 0; rf_inc = 0;

        if kzCAIPI
            gz_blips_group_rd = reshape(gz_blips_group(ind_arm,:,:), Nz/Nz_shot, Nz_shot);
        else
            gz_blips_group_rd = gz_blips_group;
        end

        seq.addBlock(rf90, mr.makeDelay(delay_1_t2prep));
        seq.addBlock(t2prep_pulse, mr.makeDelay(delay_2_t2prep));
        seq.addBlock(t2prep_pulse, mr.makeDelay(delay_2_t2prep));
        seq.addBlock(t2prep_pulse, mr.makeDelay(delay_2_t2prep));
        seq.addBlock(t2prep_pulse, mr.makeDelay(delay_3_t2prep));
        seq.addBlock(rf90_180PhaseOffset);
        seq.addBlock(gslSp_t2prep);

        phi = (ord_arm(1,ind_arm)-1)*rot_angle;
        if enable_FS; seq.addBlock(gp_r,gp_p,gp_s); seq.addBlock(rf_fs,gn_r,gn_p,gn_s); end
        [rf_phase,rf_inc] = addAcq_FS(seq, nETL, rf, gz, gzReph, adc, rfSpoilingInc, ...
            rf_phase, rf_inc, gz_blips_group_rd, no_blip_delay, gx_spi_out, gy_spi_out, ...
            phi, delay_TE_dixon, delay_TRinner, gx_spoil, gy_spoil, gz_spoil, ...
            ord_group(g), isDummy, gp_r, gp_p, gp_s, rf_fs, gn_r, gn_p, gn_s, num_part, enable_FS);
        seq.addBlock(mr.makeDelay(delT_M3_M4));

        seq.addBlock(IRprep_pulse);
        seq.addBlock(gslSp_IRprep, mr.makeDelay(delay_IRprep));

        for ind_acq = 2:5
            phi = (ord_arm(ind_acq,ind_arm)-1)*rot_angle;
            if enable_FS; seq.addBlock(gp_r,gp_p,gp_s); seq.addBlock(rf_fs,gn_r,gn_p,gn_s); end
            [rf_phase,rf_inc] = addAcq_FS(seq, nETL, rf, gz, gzReph, adc, rfSpoilingInc, ...
                rf_phase, rf_inc, gz_blips_group_rd, no_blip_delay, gx_spi_out, gy_spi_out, ...
                phi, delay_TE_dixon, delay_TRinner, gx_spoil, gy_spoil, gz_spoil, ...
                ord_group(g), isDummy, gp_r, gp_p, gp_s, rf_fs, gn_r, gn_p, gn_s, num_part, enable_FS);
            if ind_acq < 5
                seq.addBlock(mr.makeDelay(delay_TRouter));
            else
                if delT_M13_2end > 0; seq.addBlock(mr.makeDelay(delT_M13_2end)); end
                if enable_NAV && nav_after_contrast(5)
                    seq.addBlock(mr.makeDelay(delay_before_vNav));
                    [rf_phase,rf_inc] = addSpiralVNav(seq, rf, gz, gzReph, ...
                        adc_nav_template, rfSpoilingInc, rf_phase, rf_inc, ...
                        gzEnc_nav, gx_nav, gy_nav, ...
                        gxSpoil_nav, gySpoil_nav, gzSpoil_nav, delayTR_nav, isDummy);
                    seq.addBlock(mr.makeDelay(delay_after_vNav));
                end
            end
        end
    end
end

%% Sequence validation and export
fprintf('\nRunning timing check...\n');
[ok, error_report] = seq.checkTiming;
if ok
    fprintf('Timing check passed.\n');
else
    fprintf('Timing check FAILED:\n');
    fprintf([error_report{:}]);
    fprintf('\n');
end

% Sequence definitions
seq.setDefinition('FOV',              fov);
seq.setDefinition('Matrix',           N);
seq.setDefinition('Ninterl',          N_interl);
seq.setDefinition('L_interl',         size(spiral_grad_shape,2));
seq.setDefinition('slice_order',      slice_indices_icd_mid);
seq.setDefinition('n_rep',            n_rep);
seq.setDefinition('n_echo',           length(TE));
seq.setDefinition('nETL',             nETL);
seq.setDefinition('Nz_shot',          Nz_shot);
seq.setDefinition('n_readout',        5);
seq.setDefinition('turbo_factor',     nETL);
seq.setDefinition('ESP',              TR_FLASH);
seq.setDefinition('TR',               T_QALAS_cycle);
seq.setDefinition('flip_angle',       flip_angle);
seq.setDefinition('gap_between_readouts', gap_between_readouts);
seq.setDefinition('Nz_nav',           Nz_nav);
seq.setDefinition('TR_nav_shot',      TR_nav_shot);
seq.setDefinition('TE_nav',           TE_nav);

if save_seq
    fprintf('\nSaving %s.seq ...\n', seq_name_to_save);
    seq.write([seq_name_to_save '.seq']);
    fprintf('Done.\n');
end

% Scan time
N_cycles   = N_interl * Nz_shot * n_rep + 1;  % +1 for dummy
TA_QALAS   = T_QALAS_cycle * N_cycles;
TR_ref_actual = Texc_ref + TslabReph_ref + Tpre_ref + delayTE_ref + TreadBlock_ref + Tpost_ref + delayTR_ref;
TA_ref     = TR_ref_actual * (Ndummy_ref + Ny_ref*Nz_ref);
TA         = TA_ref + TA_QALAS;

fprintf('\n<strong>QALAS cycle: %.3f ms</strong>\n', T_QALAS_cycle*1e3);
fprintf('Reference scan: %.2f s\n', TA_ref);
fprintf('QALAS + vNav:   %.2f min\n', TA_QALAS/60);
fprintf('<strong>Total: %.2f min</strong>\n', TA/60);

% Optional sequence visualization
if see_seqeyes
    addpath('/autofs/cluster/berkin/xingwang/share/seqeyes');
    seqeyes(seq);
end

% K-space trajectory calculation
warning('OFF','mr:restoreShape');
[ktraj_adc,t_adc,ktraj,t_ktraj,t_excitation,t_refocusing] = seq.calculateKspacePP();

% Peripheral nerve stimulation check
if cal_pns
    addpath('/autofs/cluster/berkin/unay/ssl_mrinufft/ZS-SSL-NUFFT/matlab/sequences/SpiralQalas/Yuting/safe_pns_prediction')
    fprintf('\nCalculating PNS...\n');
    warning('OFF','mr:restoreShape');
    seq.calcPNS('/autofs/cluster/berkin/yuting/MATLAB/lib/asc/MP_GPA_K2309_2250V_951A_AS82.asc');
end

%% Local helper functions

function [rf_phase, rf_inc] = addAcq_FS(seq, nETL, rf, gz, gzReph, adc, ...
    rfSpoilingInc, rf_phase, rf_inc, gz_blips_group, no_blip_delay, ...
    gx_spi_out, gy_spi_out, phi, delay_TE_dixon, delay_TRinner, ...
    gx_spoil, gy_spoil, gz_spoil, group, isDummy, ...
    gp_r, gp_p, gp_s, rf_fs, gn_r, gn_p, gn_s, num_part, UseFatSat)

for ind_slc = 1:nETL/num_part/length(delay_TE_dixon)
    for ii = 1:num_part
        for ee = 1:length(delay_TE_dixon)
            ind_etl = (ind_slc-1)*num_part*length(delay_TE_dixon) + ii*length(delay_TE_dixon) + ee;
            phi_part = phi(ii);

            rf.phaseOffset  = rf_phase/180*pi;
            adc.phaseOffset = rf_phase/180*pi;
            rf_inc   = mod(rf_inc   + rfSpoilingInc, 360.0);
            rf_phase = mod(rf_phase + rf_inc,         360.0);

            seq.addBlock(rf, gz);
            seq.addBlock(gzReph);

            if isempty(gz_blips_group(ind_slc,group).type)
                seq.addBlock(no_blip_delay);
            else
                seq.addBlock(gz_blips_group(ind_slc,group));
            end

            if isDummy
                seq.addBlock(mr.makeDelay(delay_TE_dixon(ee)));
                seq.addBlock(mr.rotate('z', phi_part, gx_spi_out, gy_spi_out, ...
                    mr.makeDelay(mr.calcDuration(adc))));
            else
                seq.addBlock(mr.makeDelay(delay_TE_dixon(ee)));
                seq.addBlock(mr.rotate('z', phi_part, gx_spi_out, gy_spi_out, adc));
            end

            seq.addBlock(mr.rotate('z', phi_part, gx_spoil, gy_spoil, gz_spoil));
            seq.addBlock(mr.makeDelay(delay_TRinner(ee)));

            if ind_etl == ceil(nETL/2) && UseFatSat
                seq.addBlock(gp_r, gp_p, gp_s);
                seq.addBlock(rf_fs, gn_r, gn_p, gn_s);
            end
        end
    end
end
end % addAcq_FS

function [rf_phase, rf_inc] = addSpiralVNav(seq, rf, gz, gzReph, ...
    adc_nav_template, rfSpoilingInc, rf_phase, rf_inc, ...
    gzEnc_nav, gx_nav, gy_nav, ...
    gxSpoil_nav, gySpoil_nav, gzSpoil_nav, delayTR_nav, isDummy)
% Apply ADC phase offsets to a local copy for each navigator shot.
% Return the updated RF spoiling phase for continuity.

Nz_nav = length(gzEnc_nav);

for iZ_nav = 1:Nz_nav

    % Apply RF spoiling to a local copy of rf
    rf_shot = rf;
    rf_shot.phaseOffset = rf_phase * pi/180;

    rf_inc   = mod(rf_inc   + rfSpoilingInc, 360);
    rf_phase = mod(rf_phase + rf_inc,         360);

    seq.addBlock(rf_shot, gz);
    seq.addBlock(gzReph);

    % kz encoding (trapezoid or delay for kz=0)
    seq.addBlock(gzEnc_nav{iZ_nav});

    % Spiral readout
    if isDummy
        seq.addBlock(gx_nav, gy_nav);
    else
        % Apply phase to a local copy of the ADC template
        adc_shot = adc_nav_template;
        adc_shot.phaseOffset = rf_shot.phaseOffset;
        seq.addBlock(gx_nav, gy_nav, adc_shot);
    end

    % Post-readout spoiling
    seq.addBlock(gxSpoil_nav, gySpoil_nav, gzSpoil_nav);

    if delayTR_nav > 0
        seq.addBlock(mr.makeDelay(delayTR_nav));
    end
end

end % addSpiralVNav
