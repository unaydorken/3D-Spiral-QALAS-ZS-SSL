function [gz_blips, tmpj] = prepare_gz_blips_group_CAIPI_Rz2(Nz,Rz,N_group,kz_co,deltak,sys,interleave_id)
% Center-out kz encoding with layered-CAIPI ordering and Rz undersampling.
% The sampled kz residue alternates with interleave_id.

    tmp = Nz;
    if mod(tmp,2) == 1; tmp = tmp + 1; end
    for i = 1:tmp
        if mod(Nz,2) == 1
            area = -(deltak(3)*(Nz/2)) + deltak(3)*(i-1) + deltak(3)/2;
        else
            area = -(deltak(3)*(Nz/2)) + deltak(3)*(i-1);
        end
        if area ~= 0
            if i == 1
                gz_blips(i) = mr.makeTrapezoid('z',sys,'Area',area);
            else
                gz_blips(i) = mr.makeTrapezoid('z',sys,'Area',area, ...
                    'Duration',mr.calcDuration(gz_blips(1)));
            end
        end
    end

    if kz_co == 1
        tmpj = [];
        if mod(N_group,2) == 0
            g_beg = -floor(N_group/2)+1;
        else
            g_beg = -floor(N_group/2);
        end
        for g = g_beg:floor(N_group/2)
            center = round(Nz/2)+g;
            j = center;
            tmpj = [tmpj j];
            for i = 1:Nz/N_group/2
                for updown = 1:2
                    if updown == 2
                        j = center-i*N_group;
                    else
                        j = center+i*N_group;
                    end
                    if j <= Nz && j >= 1
                        tmpj = [tmpj j];
                    end
                end
            end
        end

        N_keep = 12;
        ringsPerLayer = 4;
        tmpj = layered_caipi_from_centerout(tmpj,Nz,N_keep,ringsPerLayer,interleave_id);
    else
        tmpj = 1:Nz;
    end

    % Select one of the Rz complementary kz subsets per spiral interleaf.
    kz_offset = mod(interleave_id-1,Rz);
    sampled_indices = (1+kz_offset):Rz:Nz;
    tmpj = tmpj(ismember(tmpj,sampled_indices));
    assert(numel(tmpj) == Nz/Rz, 'Unexpected number of kz partitions.');
    gz_blips = gz_blips(tmpj);
end

function tmpj2 = layered_caipi_from_centerout(tmpj,Nz,N_keep,ringsPerLayer,tIdx)
    N = numel(tmpj);
    N_keep = min(N_keep,N);
    c = floor(Nz/2)+1;
    head = tmpj(1:N_keep);
    outer = tmpj(N_keep+1:end);
    if isempty(outer)
        tmpj2 = tmpj;
        return;
    end
    r = abs(outer-c);
    ringVals = unique(r(:).');
    ringVals = sort(ringVals);
    ringVals(ringVals==0) = [];
    nLayers = ceil(numel(ringVals)/ringsPerLayer);
    outer2 = [];
    for L = 1:nLayers
        rs = (L-1)*ringsPerLayer+1;
        re = min(L*ringsPerLayer,numel(ringVals));
        ringSet = ringVals(rs:re);
        layerList = outer(ismember(r,ringSet));
        M = numel(layerList);
        if M <= 1
            outer2 = [outer2 layerList];
            continue;
        end
        s = floor(M/2)+1;
        while gcd(s,M) ~= 1
            s = s+1;
            if s > M-1
                s = 1;
                break;
            end
        end
        shift = mod(tIdx*s,M);
        outer2 = [outer2 circshift(layerList,[0 shift])];
    end
    tmpj2 = [head outer2];
end
