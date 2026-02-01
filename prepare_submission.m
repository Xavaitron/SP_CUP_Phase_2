function prepare_submission()
    %% PREPARE_SUBMISSION - Creates submission folder structure for SP Cup
    %
    %  This script organizes evaluation outputs into the required 
    %  SP Cup submission format as per Section 2.5 of the documentation.
    %
    %  Required folder structure:
    %    /Submission/
    %       /Task1_Anechoic/
    %          Task1_Anechoic_5dB.mat
    %          target_signal.wav
    %          interference_signal1.wav (Female)
    %          interference_signal2.wav (Music)
    %          interference_signal3.wav (Noise)
    %          processed_signal.wav
    %          process_task1.m
    %       /Task2_Reverberant/
    %          Task2_Reverberant_5dB.mat
    %          target_signal.wav
    %          interference_signal1.wav (Female)
    %          interference_signal2.wav (Music)
    %          interference_signal3.wav (Noise)
    %          processed_signal.wav
    %          process_task2.m
    
    clc; close all;
    fprintf('======================================\n');
    fprintf('  SP CUP SUBMISSION FOLDER GENERATOR\n');
    fprintf('======================================\n\n');

    %% CONFIGURATION
    % Source: Evaluation output folders from Python inference
    evalAnechoicDir = './Model Inference/evaluation_anechoic';
    evalReverbDir   = './Model Inference/evaluation_reverb';
    
    % Source: Dataset folders (for RIR and spatial data)
    datasetAnechoicDir = './Train_Dataset/a';
    datasetReverbDir   = './Train_Dataset/r';
    
    % Output: Submission folder
    submissionDir = './Submission';
    task1Dir = fullfile(submissionDir, 'Task1_Anechoic');
    task2Dir = fullfile(submissionDir, 'Task2_Reverberant');
    
    % Audio parameters
    fs = 16000;  % Sample rate

    %% CREATE FOLDER STRUCTURE
    fprintf('1. Creating folder structure...\n');
    
    if ~exist(submissionDir, 'dir'), mkdir(submissionDir); end
    if ~exist(task1Dir, 'dir'), mkdir(task1Dir); end
    if ~exist(task2Dir, 'dir'), mkdir(task2Dir); end
    
    fprintf('   - Created: %s\n', task1Dir);
    fprintf('   - Created: %s\n', task2Dir);

    %% TASK 1: ANECHOIC
    fprintf('\n2. Processing Task 1 (Anechoic)...\n');
    processTask(evalAnechoicDir, datasetAnechoicDir, task1Dir, 'Task1', 'Anechoic', fs);
    
    %% TASK 2: REVERBERANT
    fprintf('\n3. Processing Task 2 (Reverberant)...\n');
    processTask(evalReverbDir, datasetReverbDir, task2Dir, 'Task2', 'Reverberant', fs);

    %% COPY PROCESSING SCRIPTS AND MODEL FILES
    fprintf('\n4. Copying processing scripts and model files...\n');
    
    % Model inference directory
    modelInferenceDir = './Model Inference';
    
    % Task 1 - Anechoic: Copy model and create Python script
    copyfile(fullfile(modelInferenceDir, 'anechoic_Conformer.pth'), task1Dir);
    fprintf('   - Copied: anechoic_Conformer.pth\n');
    createProcessTaskScript(modelInferenceDir, task1Dir, 'process_task1.py', 'anechoic');
    fprintf('   - Created: process_task1.py\n');
    
    % Task 2 - Reverberant: Copy model and create Python script
    copyfile(fullfile(modelInferenceDir, 'reverb_Conformer.pth'), task2Dir);
    fprintf('   - Copied: reverb_Conformer.pth\n');
    createProcessTaskScript(modelInferenceDir, task2Dir, 'process_task2.py', 'reverb');
    fprintf('   - Created: process_task2.py\n');
    
    %% DONE
    fprintf('\n======================================\n');
    fprintf('  SUBMISSION FOLDER READY!\n');
    fprintf('======================================\n');
    fprintf('\nLocation: %s\n', fullfile(pwd, submissionDir));
    fprintf('\nTo submit: Zip the entire Submission folder.\n');
end


%% ========================================================================
function processTask(evalDir, datasetDir, taskDir, taskName, condition, fs)
    % Process evaluation outputs into submission format
    % Now loads RIR and spatial params from dataset generation
    
    % --- 1. Copy BEST_OVERALL as primary submission ---
    targetSrc   = fullfile(evalDir, 'BEST_OVERALL_target.wav');
    mixtureSrc  = fullfile(evalDir, 'BEST_OVERALL_mixture.wav');
    outputSrc   = fullfile(evalDir, 'BEST_OVERALL_output.wav');
    
    % Check if files exist
    if ~exist(targetSrc, 'file')
        error('Missing file: %s\nRun Python evaluation first!', targetSrc);
    end
    
    % Read audio files (BEST_OVERALL = primary submission)
    [target_signal, ~] = audioread(targetSrc);
    [mixture_signal, ~] = audioread(mixtureSrc);
    [processed_signal, ~] = audioread(outputSrc);
    
    fprintf('   - Read BEST_OVERALL files from: %s\n', evalDir);
    
    % --- 2. Load source (target) signals for each interference category ---
    targetFemale = fullfile(evalDir, 'BEST_MALE__FEMALE_target.wav');
    targetMusic  = fullfile(evalDir, 'BEST_MALE__MUSIC_target.wav');
    targetNoise  = fullfile(evalDir, 'BEST_MALE__NOISE_target.wav');
    
    % Read source signals (one for each interference type)
    source1 = audioread(targetFemale);  % Source when paired with Female interference
    source2 = audioread(targetMusic);   % Source when paired with Music interference
    source3 = audioread(targetNoise);   % Source when paired with Noise interference
    
    % --- 3. Load interference signals by category ---
    interfFemale = fullfile(evalDir, 'BEST_MALE__FEMALE_interference.wav');
    interfMusic  = fullfile(evalDir, 'BEST_MALE__MUSIC_interference.wav');
    interfNoise  = fullfile(evalDir, 'BEST_MALE__NOISE_interference.wav');
    
    % Read interference signals
    interf1 = audioread(interfFemale);
    interf2 = audioread(interfMusic);
    interf3 = audioread(interfNoise);
    
    % --- 4. Load processed (output) signals for each category ---
    outputFemale = fullfile(evalDir, 'BEST_MALE__FEMALE_output.wav');
    outputMusic  = fullfile(evalDir, 'BEST_MALE__MUSIC_output.wav');
    outputNoise  = fullfile(evalDir, 'BEST_MALE__NOISE_output.wav');
    
    % Read processed signals
    processed1 = audioread(outputFemale);
    processed2 = audioread(outputMusic);
    processed3 = audioread(outputNoise);
    
    % --- 5. Load mixture signals for each category ---
    mixtureFemale = fullfile(evalDir, 'BEST_MALE__FEMALE_mixture.wav');
    mixtureMusic  = fullfile(evalDir, 'BEST_MALE__MUSIC_mixture.wav');
    mixtureNoise  = fullfile(evalDir, 'BEST_MALE__NOISE_mixture.wav');
    
    % Read mixture signals
    mixture1 = audioread(mixtureFemale);
    mixture2 = audioread(mixtureMusic);
    mixture3 = audioread(mixtureNoise);
    
    % Primary source and interference for .mat file (from BEST_OVERALL = Female category)
    interference_signal = interf1;
    
    % --- 6. Save audio files with correct naming ---
    audiowrite(fullfile(taskDir, 'target_signal.wav'), target_signal, fs);
    audiowrite(fullfile(taskDir, 'processed_signal.wav'), processed_signal, fs);
    
    % Save all 3 source signals
    audiowrite(fullfile(taskDir, 'source_signal1.wav'), source1, fs);
    audiowrite(fullfile(taskDir, 'source_signal2.wav'), source2, fs);
    audiowrite(fullfile(taskDir, 'source_signal3.wav'), source3, fs);
    
    % Save all 3 interference signals
    audiowrite(fullfile(taskDir, 'interference_signal1.wav'), interf1, fs);
    audiowrite(fullfile(taskDir, 'interference_signal2.wav'), interf2, fs);
    audiowrite(fullfile(taskDir, 'interference_signal3.wav'), interf3, fs);
    
    % Save all 3 processed signals
    audiowrite(fullfile(taskDir, 'processed_signal1.wav'), processed1, fs);
    audiowrite(fullfile(taskDir, 'processed_signal2.wav'), processed2, fs);
    audiowrite(fullfile(taskDir, 'processed_signal3.wav'), processed3, fs);
    
    % Save all 3 mixture signals (for inference input)
    audiowrite(fullfile(taskDir, 'mixture_signal1.wav'), mixture1, fs);
    audiowrite(fullfile(taskDir, 'mixture_signal2.wav'), mixture2, fs);
    audiowrite(fullfile(taskDir, 'mixture_signal3.wav'), mixture3, fs);
    
    fprintf('   - Saved audio files:\n');
    fprintf('     * target_signal.wav, processed_signal.wav (BEST_OVERALL)\n');
    fprintf('     * source_signal1/2/3.wav (Female/Music/Noise pairs)\n');
    fprintf('     * interference_signal1/2/3.wav (Female/Music/Noise)\n');
    fprintf('     * processed_signal1/2/3.wav (Female/Music/Noise outputs)\n');
    fprintf('     * mixture_signal1/2/3.wav (Female/Music/Noise inputs)\n');
    
    % --- 4. Load RIR and spatial data from dataset generation ---
    % Find first sample folder with spatial_data.mat
    sampleDirs = dir(fullfile(datasetDir, 'sample_*'));
    if isempty(sampleDirs)
        warning('No sample folders found in %s. Using fallback RIR.', datasetDir);
        % Fallback: generate placeholder RIRs for both target and interference
        if strcmp(condition, 'Anechoic')
            rir_target = [1, zeros(1, 4095); 1, zeros(1, 4095)];  % 2x4096 impulse
            rir_interf = [1, zeros(1, 4095); 1, zeros(1, 4095)];  % 2x4096 impulse
            rt60 = 0.0;
        else
            rt60 = 0.5;
            t = (0:4095) / fs;
            decay = exp(-3 * t / rt60);
            rir_target = [randn(1, 4096) .* decay; randn(1, 4096) .* decay];
            rir_target = rir_target / max(abs(rir_target(:)));
            rir_interf = [randn(1, 4096) .* decay; randn(1, 4096) .* decay];
            rir_interf = rir_interf / max(abs(rir_interf(:)));
        end
        params.sampling_rate = fs;
        params.mic_positions = [2.41, 2.45, 1.5; 2.49, 2.45, 1.5];
        params.array_spacing = 0.08;
        params.source_azimuth = 90;
        params.interferer_azimuth = 40;
        params.source_height = 1.5;
        params.SNR_dB = 5;
        params.SIR_dB = 0;
        params.RT60 = rt60;
        params.room_dimensions = [4.9, 4.9, 4.9];
    else
        % Load from first sample's spatial_data.mat
        spatialDataPath = fullfile(datasetDir, sampleDirs(1).name, 'spatial_data.mat');
        if exist(spatialDataPath, 'file')
            spatialData = load(spatialDataPath);
            % Load both RIRs (target and interference)
            rir_target = spatialData.rir_target;  % Full 2xN RIR matrix
            rir_interf = spatialData.rir_interf;  % Full 2xN RIR matrix
            params = spatialData.params;
            fprintf('   - Loaded RIRs (target + interf) and params from: %s\n', spatialDataPath);
        else
            error('spatial_data.mat not found in %s. Run dataset generation first!', sampleDirs(1).name);
        end
    end
    
    % --- 5. Read metrics from JSON ---
    metricsFile = fullfile(evalDir, 'metrics.json');
    if exist(metricsFile, 'file')
        metricsText = fileread(metricsFile);
        metricsJson = jsondecode(metricsText);
        
        metrics.OSINR = metricsJson.best_overall.sisdr;  % Using SI-SDR as OSINR
        metrics.PESQ  = metricsJson.best_overall.pesq;
        metrics.STOI  = metricsJson.best_overall.stoi;
    else
        % Defaults if no metrics file
        metrics.OSINR = 0;
        metrics.PESQ  = 0;
        metrics.STOI  = 0;
    end
    
    fprintf('   - Metrics: OSINR=%.2f dB, PESQ=%.2f, STOI=%.2f\n', ...
            metrics.OSINR, metrics.PESQ, metrics.STOI);
    
    % Note: params struct is loaded from spatial_data.mat (step 4)
    
    % --- 6. Save .mat file ---
    matFileName = sprintf('%s_%s_5dB.mat', taskName, condition);
    matFilePath = fullfile(taskDir, matFileName);
    
    save(matFilePath, 'target_signal', 'interference_signal', 'mixture_signal', ...
         'rir_target', 'rir_interf', 'processed_signal', 'metrics', 'params', '-v7.3');
    
    fprintf('   - Created: %s\n', matFileName);
end


%% ========================================================================
function createProcessTaskScript(modelInferenceDir, taskDir, filename, condition)
    % Create process_task script by modifying inference_Conformer.py
    % Takes sample number (1, 2, or 3) instead of input file path
    
    % Read the original inference script
    srcPath = fullfile(modelInferenceDir, 'inference_Conformer.py');
    srcContent = fileread(srcPath);
    
    % Determine model file and task info
    modelFile = [condition '_Conformer.pth'];
    if strcmp(condition, 'anechoic')
        taskNum = '1';
        conditionName = 'Anechoic';
    else
        taskNum = '2';
        conditionName = 'Reverberant';
    end
    
    % Create global config block to insert after imports
    globalConfig = sprintf([...
        '\n# ==========================================\n' ...
        '# CONFIGURATION (Task %s - %s)\n' ...
        '# ==========================================\n' ...
        'SAMPLE_RATE = 16000\n' ...
        'FIXED_DURATION = 3.0  # seconds\n' ...
        'FIXED_SAMPLES = int(SAMPLE_RATE * FIXED_DURATION)\n' ...
        'TARGET_ANGLE = 90  # degrees\n' ...
        '\n' ...
        '# Script directory and paths\n' ...
        'SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))\n' ...
        'MODEL_PATH = os.path.join(SCRIPT_DIR, "%s")\n' ...
        '\n' ...
        '# Sample mapping: 1=Female, 2=Music, 3=Noise\n' ...
        'def get_input_path(sample_num):\n' ...
        '    return os.path.join(SCRIPT_DIR, f"mixture_signal{sample_num}.wav")\n' ...
        '\n' ...
        'def get_output_path(sample_num):\n' ...
        '    return os.path.join(SCRIPT_DIR, f"output_signal{sample_num}.wav")\n' ...
        '\n'], taskNum, conditionName, modelFile);
    
    % Replace the main() function to take sample number
    newMain = sprintf([...
        'def main():\n' ...
        '    parser = argparse.ArgumentParser(description="Task %s - %s Source Separation")\n' ...
        '    parser.add_argument("--sample", "-s", type=int, required=True, choices=[1, 2, 3],\n' ...
        '                        help="Sample number (1=Female, 2=Music, 3=Noise)")\n' ...
        '    parser.add_argument("--angle", "-a", type=float, default=TARGET_ANGLE,\n' ...
        '                        help="Target angle (0-180 degrees)")\n' ...
        '    parser.add_argument("--device", "-d", type=str, default="cpu",\n' ...
        '                        help="Device (cpu or cuda)")\n' ...
        '    args = parser.parse_args()\n' ...
        '    \n' ...
        '    # Get paths from global config\n' ...
        '    input_file = get_input_path(args.sample)\n' ...
        '    output_file = get_output_path(args.sample)\n' ...
        '    \n' ...
        '    device = torch.device(args.device)\n' ...
        '    print(f"Task %s - %s Source Separation")\n' ...
        '    print(f"Sample: {args.sample}")\n' ...
        '    print(f"Input: {input_file}")\n' ...
        '    print(f"Output: {output_file}")\n' ...
        '    print(f"Target angle: {args.angle} degrees")\n' ...
        '    print(f"Model: {MODEL_PATH}")\n' ...
        '    print(f"Device: {device}")\n' ...
        '    \n' ...
        '    # Load model\n' ...
        '    model = DCCRNConformer(n_fft=512, hop_length=128).to(device)\n' ...
        '    try:\n' ...
        '        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)\n' ...
        '        model.load_state_dict(state_dict)\n' ...
        '        print("Model loaded successfully")\n' ...
        '    except FileNotFoundError:\n' ...
        '        print("Error: Model file not found!")\n' ...
        '        return\n' ...
        '    model.eval()\n' ...
        '    \n' ...
        '    # Load audio\n' ...
        '    print("Loading audio...")\n' ...
        '    waveform = load_audio(input_file)\n' ...
        '    original_len = waveform.shape[-1]\n' ...
        '    print(f"Input duration: {original_len / SAMPLE_RATE:.2f}s")\n' ...
        '    \n' ...
        '    # Pad/trim to fixed size\n' ...
        '    if original_len > FIXED_SAMPLES:\n' ...
        '        waveform = waveform[:, :FIXED_SAMPLES]\n' ...
        '    elif original_len < FIXED_SAMPLES:\n' ...
        '        waveform = F.pad(waveform, (0, FIXED_SAMPLES - original_len))\n' ...
        '    \n' ...
        '    input_rms = torch.sqrt(torch.mean(waveform ** 2))\n' ...
        '    waveform = waveform.unsqueeze(0).to(device)\n' ...
        '    angle_tensor = torch.tensor([[args.angle]], dtype=torch.float32).to(device)\n' ...
        '    \n' ...
        '    # Inference\n' ...
        '    print("Processing...")\n' ...
        '    with torch.no_grad():\n' ...
        '        output = model(waveform, angle_tensor)\n' ...
        '    \n' ...
        '    output_len = min(original_len, FIXED_SAMPLES)\n' ...
        '    output = output[:, :output_len]\n' ...
        '    \n' ...
        '    # Match power\n' ...
        '    output_rms = torch.sqrt(torch.mean(output.cpu() ** 2))\n' ...
        '    if output_rms > 1e-8:\n' ...
        '        output = output.cpu() * (input_rms / output_rms)\n' ...
        '    else:\n' ...
        '        output = output.cpu()\n' ...
        '    \n' ...
        '    torchaudio.save(output_file, output, SAMPLE_RATE)\n' ...
        '    print(f"Saved: {output_file}")\n' ...
        '    print(f"Output duration: {output.shape[-1] / SAMPLE_RATE:.2f}s")\n' ...
        '    print("Processing complete!")\n'], ...
        taskNum, conditionName, taskNum, conditionName);
    
    % Add 'import os' after 'import math'
    srcContent = strrep(srcContent, 'import math', 'import math\nimport os');
    
    % Find where to insert config (after imports, before model classes)
    modelStart = strfind(srcContent, '# ==========================================');
    if ~isempty(modelStart)
        srcContent = [srcContent(1:modelStart(1)-1), globalConfig, srcContent(modelStart(1):end)];
    end
    
    % Remove old SAMPLE_RATE, FIXED_DURATION, FIXED_SAMPLES (we added new ones)
    srcContent = strrep(srcContent, sprintf('SAMPLE_RATE = 16000\r\nFIXED_DURATION = 3.0  # seconds - model''s expected input size\r\nFIXED_SAMPLES = int(SAMPLE_RATE * FIXED_DURATION)'), '');
    srcContent = strrep(srcContent, sprintf('SAMPLE_RATE = 16000\nFIXED_DURATION = 3.0  # seconds - model''s expected input size\nFIXED_SAMPLES = int(SAMPLE_RATE * FIXED_DURATION)'), '');
    
    % Find and replace the main() function
    mainStart = strfind(srcContent, 'def main():');
    ifNameStart = strfind(srcContent, 'if __name__ == "__main__":');
    
    if ~isempty(mainStart) && ~isempty(ifNameStart)
        newContent = [srcContent(1:mainStart(end)-1), newMain, char(10), srcContent(ifNameStart(end):end)];
    else
        newContent = srcContent;
    end
    
    % Update docstring at the top
    oldDocstring = 'Run command: python inference_Conformer.py --input <input_wav> --angle <target_angle> --output <output_wav>';
    newDocstring = sprintf('Usage: python %s --sample <1/2/3> [--angle <degrees>] [--device <cpu/cuda>]', filename);
    newContent = strrep(newContent, oldDocstring, newDocstring);
    
    % Write the modified script
    filepath = fullfile(taskDir, filename);
    fid = fopen(filepath, 'w');
    fprintf(fid, '%s', newContent);
    fclose(fid);
end

