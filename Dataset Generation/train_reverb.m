function train_reverb()
    clc; close all;


    %% 1. CONFIGURATION
    
    % --- Paths (Update these if needed) ---
    maleDir   = "D:\Dataset\Male";           
    femaleDir = "D:\Dataset\Female";         
    noiseDir  = "C:\Users\ironp\Downloads\SP Cup\matlab_stuff\MUSAN\noise";    
    musicDir  = "C:\Users\ironp\Downloads\SP Cup\matlab_stuff\MUSAN\music";    
    outputDatasetDir = '../Train_Dataset/reverb';
    
    % --- Generation Settings ---
    numSamplesToGenerate = 150000;
    desiredLength_s      = 4.0;
    fs                   = 16000;
    desiredLength_samples = floor(desiredLength_s * fs);
    
    % --- Acoustic Settings ---
    sir_dB            = 0;     % Signal-to-Interference Ratio
    snr_dB            = 5;     % Signal-to-Noise Ratio (sensor noise)
    min_angle_sep_deg = 15;    % Min separation between sources
    source_radius_m   = 1.0;
    roomDimensions    = [4.9, 4.9, 4.9];
    
    % Mic Array (2 Mics)
    micPositions      = [2.41, 2.45, 1.5;   
                         2.49, 2.45, 1.5];
    micCenter         = mean(micPositions, 1);
    
    c     = 343;   % Speed of sound
    beta  = 0.5;   % RT60
    n_rir = 4096;  % RIR Length


    if ~exist(outputDatasetDir, 'dir'), mkdir(outputDatasetDir); end


    %% 2. FILE SCANNING & MEMORY OPTIMIZATION
    fprintf('1. Scanning Audio Files...\n');
    
    rawMale   = getFiles(maleDir);
    rawFemale = getFiles(femaleDir);
    rawNoise  = getFiles(noiseDir);
    rawMusic  = getFiles(musicDir);


    if isempty(rawMale), error('No Male files found! Check paths.'); end


    % --- CRITICAL FIX: Use Constants to prevent RAM Crash ---
    fprintf('   - Storing file lists in shared memory...\n');
    cMale   = parallel.pool.Constant(rawMale);
    cFemale = parallel.pool.Constant(rawFemale);
    cNoise  = parallel.pool.Constant(rawNoise);
    cMusic  = parallel.pool.Constant(rawMusic);


    %% 3. SAFE PARALLEL POOL SETUP
    % Limit workers to 4 to prevent freezing. Increase only if you have >32GB RAM.
    SAFE_WORKERS = 4; 
    poolObj = gcp('nocreate');
    if isempty(poolObj)
        parpool('local', SAFE_WORKERS);
    elseif poolObj.NumWorkers > SAFE_WORKERS
        delete(poolObj);
        parpool('local', SAFE_WORKERS);
    end


    %% 4. RIR PRE-CALCULATION (CACHE)
    fprintf('2. Pre-calculating 3600 RIRs (This makes the loop fast)...\n');
    
    % Note: We still calculate 360 degrees to keep indexing logic simple and safe,
    % even though we only use 0-180.
    angle_step_deg = 0.1;
    num_angles = 360 / angle_step_deg; % 3600 positions
    tempCache = cell(num_angles, 1);
    
    % Pre-calculate RIRs in parallel
    parfor i_cache = 1:num_angles
        angle_deg = (i_cache - 1) * angle_step_deg;
        pos = getPositionFromAngle(angle_deg, source_radius_m, micCenter);
        tempCache{i_cache} = rir_generator(c, fs, micPositions, pos, roomDimensions, beta, n_rir);
    end
    
    % --- CRITICAL FIX: Store massive RIR cache in Constant ---
    cRirCache = parallel.pool.Constant(tempCache);
    clear tempCache; % Clear from main RAM immediately
    fprintf('   - Cache built and optimized.\n');


    %% 5. MAIN GENERATION LOOP
    fprintf('3. Starting Dataset Generation...\n');
    
    % Progress Queue
    q = parallel.pool.DataQueue;
    afterEach(q, @(x) updateProgress(x, numSamplesToGenerate));
    
    parfor i = 1:numSamplesToGenerate
        rng('shuffle'); % Ensure random variety
        
        try
            % --- A. Source Selection ---
            % Target is always Male
            mList = cMale.Value;
            targetFile = mList{randi(numel(mList))};
            
            % Interferer: 33% Female, 33% Noise, 33% Music
            roll = rand();
            if roll < 0.50
                iList = cFemale.Value;
            elseif roll < 0.75
                iList = cNoise.Value;
            else
                iList = cMusic.Value;
            end
            
            % Fallback if list is empty
            if isempty(iList), iList = cNoise.Value; end
            interfererFile = iList{randi(numel(iList))};
            
            % Process Audio
            targetSignal = preprocessAudio(targetFile, fs, desiredLength_samples);
            interfererSignal = preprocessAudio(interfererFile, fs, desiredLength_samples);
            
            % --- B. Geometry Logic ---
            targetAngle_deg = 0; 
            interfererAngle_deg = 0;
            fov_angle_deg = 0; 
            fov_width_deg = 0;
            
            while true
                % Random FOV
                fov_angle_deg = rand() * 120 + 30;
                fov_width_deg = rand() * 60 + 10;
                fov_min = fov_angle_deg - (fov_width_deg / 2);
                fov_max = fov_angle_deg + (fov_width_deg / 2);
                
                % Target inside FOV
                targetAngle_deg = rand() * fov_width_deg + fov_min;
                
                % Interferer explicitly 0 to 180 degrees
                interfererAngle_deg = rand() * 180;
                
                % Validation
                sep = abs(targetAngle_deg - interfererAngle_deg);
                
                % Keep target in front (20-160) AND ensure separation
                % Note: This range (20-160) is strictly within 0-180.
                cond1 = (targetAngle_deg >= 20) && (targetAngle_deg <= 160);
                isI_in = (interfererAngle_deg >= fov_min) && (interfererAngle_deg <= fov_max);
                
                if cond1 && ~isI_in && (sep >= min_angle_sep_deg)
                    break;
                end
            end
            
            % --- C. RIR Retrieval & Simulation ---
            % Find closest index in cache
            t_idx = floor(targetAngle_deg / angle_step_deg) + 1;
            i_idx = floor(interfererAngle_deg / angle_step_deg) + 1;
            
            % Clamp indices safely
            t_idx = max(1, min(t_idx, num_angles));
            i_idx = max(1, min(i_idx, num_angles));
            
            % Get RIRs from shared memory
            rirT = cRirCache.Value{t_idx};
            rirI = cRirCache.Value{i_idx};
            
            % Convolution
            micT = fftfilt(rirT', targetSignal);
            micI = fftfilt(rirI', interfererSignal);
            
            % Truncate to match length
            len = min(size(micT, 1), size(micI, 1));
            micT = micT(1:len, :);
            micI = micI(1:len, :);
            
            % --- D. Mixing (SIR & SNR) ---
            powT = mean(micT(:,1).^2) + 1e-12;
            powI = mean(micI(:,1).^2) + 1e-12;
            
            % Adjust Interferer to satisfy SIR = 0dB
            scale = sqrt(powT / powI) / (10^(sir_dB/20));
            micI = micI * scale;
            
            cleanMix = micT + micI;
            
            % Add White Noise (SNR = 5dB)
            powMix = mean(cleanMix(:).^2);
            noise = sqrt(powMix / (10^(snr_dB/10))) * randn(size(cleanMix));
            finalMix = cleanMix + noise;
            
            % --- E. Saving ---
            fName = sprintf('sample_%05d', i);
            subFolder = fullfile(outputDatasetDir, fName);
            if ~exist(subFolder, 'dir'), mkdir(subFolder); end
            
            % Normalize to prevent clipping
            peak = max(abs(finalMix(:)));
            if peak > 0.99, finalMix = finalMix * (0.99/peak); end
            
            audiowrite(fullfile(subFolder, 'mixture.wav'), finalMix, fs);
            audiowrite(fullfile(subFolder, 'target.wav'), targetSignal, fs);
            
            % --- SAVING INTERFERENCE (ADDED) ---
            audiowrite(fullfile(subFolder, 'interference.wav'), micI, fs);
            
            % --- F. Metadata ---
            meta = struct(); 
            meta.target_angle = targetAngle_deg;
            meta.interf_angle = interfererAngle_deg;
            meta.fov_angle    = fov_angle_deg;
            meta.fov_width    = fov_width_deg;
            meta.rt60         = beta;
            
            fid = fopen(fullfile(subFolder, 'meta.json'), 'w');
            fprintf(fid, '%s', jsonencode(meta, 'PrettyPrint', true));
            fclose(fid);
            
            % Update progress
            send(q, i);
            
        catch ME
            fprintf('Error in sample %d: %s\n', i, ME.message);
        end
    end
    fprintf('Dataset Generation Complete.\n');
end


%% --- HELPER FUNCTIONS ---


function updateProgress(~, total)
    persistent pCount
    if isempty(pCount), pCount = 0; end
    pCount = pCount + 1;
    if mod(pCount, 100) == 0
        fprintf('Progress: %d / %d\n', pCount, total);
    end
end


function files = getFiles(root)
    if strlength(root) < 1, files = {}; return; end
    f = dir(fullfile(root, '**', '*.wav'));
    f2 = dir(fullfile(root, '**', '*.flac'));
    f = [f; f2];
    files = fullfile({f.folder}, {f.name});
end


function pos = getPositionFromAngle(angle_deg, r, c)
    rad = deg2rad(angle_deg);
    pos = [c(1)+r*cos(rad), c(2)+r*sin(rad), c(3)];
end


function out = preprocessAudio(p, fs, samps)
    [x, fIn] = audioread(p);
    % Resample if needed
    if fIn ~= fs, x = resample(x, fs, fIn); end
    % Convert to mono
    if size(x, 2) > 1, x = x(:, 1); end
    
    % Pad or Crop
    if length(x) < samps
        out = [x; zeros(samps - length(x), 1)];
    else
        st = randi(length(x) - samps + 1);
        out = x(st : st + samps - 1);
    end
end
