function [gz_blips,tmpj] = prepare_gz_blips_group_CAIPI(Nz,Rz,N_group,kz_co,deltak,sys,interleave_id)
% lims = params.gen.lims;    
% Kz Blips:
    tmp = Nz;
    if mod(tmp,2) == 1; tmp = tmp + 1; end
    for i=1:tmp
        if mod(Nz,2) == 1
            area = -(deltak(3)*(Nz/2))+(deltak(3)*(i-1))+(deltak(3)/2);
        else
            area = -(deltak(3)*(Nz/2))+(deltak(3)*(i-1));
        end
        dur = ceil(2*sqrt(area/sys.maxSlew)/10e-6)*10e-6;
        if area ~= 0
            if i == 1
                gz_blips(i) = mr.makeTrapezoid('z',sys,'Area',area);
            else
                gz_blips(i) = mr.makeTrapezoid('z',sys,'Area',area,'Duration',mr.calcDuration(gz_blips(1)));
            end
        end
%         end
    end
    % if mod(floor(Nz),2) == 1
    %     gz_blips(round(Nz/2)) = [];    % Removing the empty blip...
    % else
    %     gz_blips(round(Nz/2)+1) = [];    % Removing the empty blip...
    % end
    
    % Reshuffling blips if center-out
    if kz_co == 1
        tmp = [];
        tmpj = [];
        
        if mod(N_group,2) == 0
            g_beg = -1*floor(N_group/2)+1;
        else
            g_beg = -1*floor(N_group/2);
        end
        for g = g_beg:floor(N_group/2)
            center = round(Nz/2)+g;% center slice
            j = center;
            tmp = [tmp gz_blips(j)];
            tmpj = [tmpj j];
            for i=1:Nz/N_group/2
                for updowm = 1:2
                    % while 
                    if updowm == 2  % down
                        j = center-i*N_group;
                    else % up
                        j = center+i*N_group;
                    end
                    if j<=Nz && j >=1
                        tmp = [tmp gz_blips(j)];
                        tmpj = [tmpj j];
                    end
                end
            end
        end
        % gz_blips = tmp;

        % CAIPi
        % English comment: layered-CAIPI for outer kz while keeping central samples
        % N_keep = 12;          % e.g., 8/12/16
        % ringsPerLayer = 4;    % e.g., 3~6
        N_keep = 12;          % e.g., 8/12/16
        ringsPerLayer = 4;    % e.g., 3~6

        tIdx = interleave_id; % or TR_id / contrast_id / combination
        
        tmpj = layered_caipi_from_centerout(tmpj, Nz, N_keep, ringsPerLayer, tIdx);
        gz_blips = gz_blips(tmpj);



        
    end

    % Repeting for interlaves
%     gz_blips = repmat(gz_blips',[1,params.gen.seg]);

end

function tmpj2 = layered_caipi_from_centerout(tmpj, Nz, N_keep, ringsPerLayer, tIdx)
% layered_caipi_from_centerout
% English comment:
% Keep first N_keep center-out samples fixed, then apply layered-CAIPI reordering on outer kz.
% Outer indices are grouped by |kz-center| "rings" and merged into layers of ringsPerLayer rings.
% Each layer is cyclic-shifted using a CAIPI-like coprime step that changes with tIdx.
%
% Inputs:
%   tmpj          : 1xNz center-out ordering indices (1..Nz)
%   Nz            : number of kz encodes
%   N_keep        : number of leading (central) indices to keep unchanged
%   ringsPerLayer : number of radius rings per layer (e.g., 3~6)
%   tIdx          : time index (e.g., interleave_id, TR_id, contrast_id, or combined)
%
% Output:
%   tmpj2         : reordered indices

    N = numel(tmpj);
    N_keep = min(N_keep, N);

    c = floor(Nz/2) + 1;  % English comment: kz center index

    head  = tmpj(1:N_keep);
    outer = tmpj(N_keep+1:end);

    if isempty(outer)
        tmpj2 = tmpj;
        return;
    end

    % English comment: compute ring radius r = |j - center|
    r = abs(outer - c);

    % English comment: unique ring radii in ascending order (skip r=0)
    ringVals = unique(r(:).');
    ringVals = sort(ringVals);
    ringVals(ringVals==0) = [];

    % English comment: build layers as consecutive chunks of ring radii
    nRings = numel(ringVals);
    nLayers = ceil(nRings / ringsPerLayer);

    outer2 = [];

    for L = 1:nLayers
        rs = (L-1)*ringsPerLayer + 1;
        re = min(L*ringsPerLayer, nRings);
        ringSet = ringVals(rs:re);

        % English comment: collect indices in this layer, keep original relative order from tmpj
        mask = ismember(r, ringSet);
        layerList = outer(mask);

        M = numel(layerList);
        if M <= 1
            outer2 = [outer2, layerList];
            continue;
        end

        % English comment: choose step s coprime with M (CAIPI-like)
        s = floor(M/2) + 1;
        while gcd(s, M) ~= 1
            s = s + 1;
            if s > M-1
                s = 1;
                break;
            end
        end

        shift = mod(tIdx * s, M);
        layerList2 = circshift(layerList, [0, shift]);

        outer2 = [outer2, layerList2];
    end

    tmpj2 = [head, outer2];
end
