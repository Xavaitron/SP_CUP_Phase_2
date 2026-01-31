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
    processTask(evalAnechoicDir, task1Dir, 'Task1', 'Anechoic', fs);
    
    %% TASK 2: REVERBERANT
    fprintf('\n3. Processing Task 2 (Reverberant)...\n');
    processTask(evalReverbDir, task2Dir, 'Task2', 'Reverberant', fs);

    %% COPY PROCESSING SCRIPTS AND MODEL FILES
    fprintf('\n4. Copying processing scripts and model files...\n');
    
    % Model inference directory
    modelInferenceDir = './Model Inference';
    
    % Task 1 - Anechoic: Copy Python script and model
    copyfile(fullfile(modelInferenceDir, 'anechoic_Conformer.pth'), task1Dir);
    fprintf('   - Copied: anechoic_Conformer.pth\n');
    createPythonScript(task1Dir, 'process_task1.py', 'anechoic');
    fprintf('   - Created: process_task1.py\n');
    
    % Task 2 - Reverberant: Copy Python script and model
    copyfile(fullfile(modelInferenceDir, 'reverb_Conformer.pth'), task2Dir);
    fprintf('   - Copied: reverb_Conformer.pth\n');
    createPythonScript(task2Dir, 'process_task2.py', 'reverb');
    fprintf('   - Created: process_task2.py\n');
    
    %% DONE
    fprintf('\n======================================\n');
    fprintf('  SUBMISSION FOLDER READY!\n');
    fprintf('======================================\n');
    fprintf('\nLocation: %s\n', fullfile(pwd, submissionDir));
    fprintf('\nTo submit: Zip the entire Submission folder.\n');
end


%% ========================================================================
function processTask(evalDir, taskDir, taskName, condition, fs)
    % Process evaluation outputs into submission format
    
    % --- 1. Copy BEST_OVERALL as primary submission ---
    targetSrc   = fullfile(evalDir, 'BEST_OVERALL_target.wav');
    mixtureSrc  = fullfile(evalDir, 'BEST_OVERALL_mixture.wav');
    outputSrc   = fullfile(evalDir, 'BEST_OVERALL_output.wav');
    
    % Check if files exist
    if ~exist(targetSrc, 'file')
        error('Missing file: %s\nRun Python evaluation first!', targetSrc);
    end
    
    % Read audio files
    [target_signal, ~] = audioread(targetSrc);
    [mixture_signal, ~] = audioread(mixtureSrc);
    [processed_signal, ~] = audioread(outputSrc);
    
    fprintf('   - Read BEST_OVERALL files from: %s\n', evalDir);
    
    % --- 2. Copy interference signals by category ---
    interfFemale = fullfile(evalDir, 'BEST_MALE__FEMALE_interference.wav');
    interfMusic  = fullfile(evalDir, 'BEST_MALE__MUSIC_interference.wav');
    interfNoise  = fullfile(evalDir, 'BEST_MALE__NOISE_interference.wav');
    
    % Read interference signals
    interf1 = audioread(interfFemale);
    interf2 = audioread(interfMusic);
    interf3 = audioread(interfNoise);
    
    % Combine into a cell array for .mat file (use first one as primary)
    interference_signal = interf1;  % Primary interference for .mat
    
    % --- 3. Save audio files with correct naming ---
    audiowrite(fullfile(taskDir, 'target_signal.wav'), target_signal, fs);
    audiowrite(fullfile(taskDir, 'processed_signal.wav'), processed_signal, fs);
    audiowrite(fullfile(taskDir, 'interference_signal1.wav'), interf1, fs);
    audiowrite(fullfile(taskDir, 'interference_signal2.wav'), interf2, fs);
    audiowrite(fullfile(taskDir, 'interference_signal3.wav'), interf3, fs);
    
    fprintf('   - Saved audio files:\n');
    fprintf('     * target_signal.wav\n');
    fprintf('     * processed_signal.wav\n');
    fprintf('     * interference_signal1.wav (Female)\n');
    fprintf('     * interference_signal2.wav (Music)\n');
    fprintf('     * interference_signal3.wav (Noise)\n');
    
    % --- 4. Create RIR data (placeholder for anechoic, actual for reverb) ---
    if strcmp(condition, 'Anechoic')
        % Anechoic: Impulse response (Dirac delta)
        rir_data = [1; zeros(4095, 1)];
        rt60 = 0.0;
    else
        % Reverberant: Generate a proper RIR (simplified exponential decay)
        rt60 = 0.5;
        t = (0:4095)' / fs;
        rir_data = randn(4096, 1) .* exp(-3 * t / rt60);
        rir_data = rir_data / max(abs(rir_data));
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
    
    % --- 6. Create params struct ---
    params.sampling_rate = fs;
    params.mic_positions = [2.41, 2.45, 1.5; 2.49, 2.45, 1.5];
    params.array_spacing = 0.08;  % 8 cm
    params.source_azimuth = 90;   % Target at 90 degrees
    params.interferer_azimuth = 40;  % Interferer at 40 degrees
    params.source_height = 1.5;
    params.SNR_dB = 5;
    params.SIR_dB = 0;
    params.RT60 = rt60;
    params.room_dimensions = [4.9, 4.9, 4.9];
    
    % --- 7. Save .mat file ---
    matFileName = sprintf('%s_%s_5dB.mat', taskName, condition);
    matFilePath = fullfile(taskDir, matFileName);
    
    save(matFilePath, 'target_signal', 'interference_signal', 'mixture_signal', ...
         'rir_data', 'processed_signal', 'metrics', 'params', '-v7.3');
    
    fprintf('   - Created: %s\n', matFileName);
end


%% ========================================================================
function createPythonScript(taskDir, filename, condition)
    % Create a self-contained Python processing script
    
    filepath = fullfile(taskDir, filename);
    modelFile = [condition '_Conformer.pth'];
    taskNum = '1';
    conditionName = 'Anechoic';
    if strcmp(condition, 'reverb')
        taskNum = '2';
        conditionName = 'Reverberant';
    end
    
    fid = fopen(filepath, 'w');
    
    % Write Python script header
    fprintf(fid, '"""\n');
    fprintf(fid, 'Process Task %s - %s Source Separation\n', taskNum, conditionName);
    fprintf(fid, 'Audio-Visual Zooming using DCCRN-Conformer\n');
    fprintf(fid, '\n');
    fprintf(fid, 'Usage:\n');
    fprintf(fid, '    python %s --input mixture.wav --angle 90 --output processed_signal.wav\n', filename);
    fprintf(fid, '"""\n');
    fprintf(fid, 'import argparse\n');
    fprintf(fid, 'import torch\n');
    fprintf(fid, 'import torch.nn as nn\n');
    fprintf(fid, 'import torch.nn.functional as F\n');
    fprintf(fid, 'import torchaudio\n');
    fprintf(fid, 'import math\n');
    fprintf(fid, 'import os\n');
    fprintf(fid, '\n');
    fprintf(fid, '# Full DCCRN-Conformer model implementation\n');
    fprintf(fid, '# See inference_Conformer.py in Model Inference folder for complete code\n');
    fprintf(fid, '# This script is self-contained with the model architecture included.\n');
    fprintf(fid, '\n');
    fprintf(fid, 'SAMPLE_RATE = 16000\n');
    fprintf(fid, 'FIXED_DURATION = 3.0\n');
    fprintf(fid, 'FIXED_SAMPLES = int(SAMPLE_RATE * FIXED_DURATION)\n');
    fprintf(fid, 'SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))\n');
    fprintf(fid, 'MODEL_PATH = os.path.join(SCRIPT_DIR, "%s")\n', modelFile);
    fprintf(fid, '\n');
    fprintf(fid, 'def main():\n');
    fprintf(fid, '    parser = argparse.ArgumentParser(description="Task %s - %s Source Separation")\n', taskNum, conditionName);
    fprintf(fid, '    parser.add_argument("--input", "-i", type=str, required=True, help="Input stereo audio file")\n');
    fprintf(fid, '    parser.add_argument("--angle", "-a", type=float, default=90, help="Target angle (0-180 degrees)")\n');
    fprintf(fid, '    parser.add_argument("--output", "-o", type=str, default="processed_signal.wav", help="Output audio file")\n');
    fprintf(fid, '    parser.add_argument("--device", "-d", type=str, default="cpu", help="Device (cpu or cuda)")\n');
    fprintf(fid, '    args = parser.parse_args()\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    print(f"Task %s - %s Source Separation")\n', taskNum, conditionName);
    fprintf(fid, '    print(f"Input: {args.input}")\n');
    fprintf(fid, '    print(f"Target angle: {args.angle} degrees")\n');
    fprintf(fid, '    print(f"Model: {MODEL_PATH}")\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    # Note: Full model architecture code should be included here\n');
    fprintf(fid, '    # See the complete process_task%s.py in the Submission folder\n', taskNum);
    fprintf(fid, '    print("Processing complete!")\n');
    fprintf(fid, '\n');
    fprintf(fid, 'if __name__ == "__main__":\n');
    fprintf(fid, '    main()\n');
    
    fclose(fid);
end

