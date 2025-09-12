clear;
clc;                                               
close all; % Close open windows

% Get the current working directory
currentDir = pwd;
 
% Find the parent directory
[parentDir, ~, ~] = fileparts(currentDir);

% read setup file and 
setup = readtable("setup.csv");

% Filter out the simulations that already exist in the setup file
dataDir = fullfile(parentDir, "Data");
fileList = dir(fullfile(dataDir, "*.csv")); % Get all CSV files

RunIDs = [];
pattern = "RunID-(\d+)-fseed"; %extract RunID

for i = 1:length(fileList)
    fileName = fileList(i).name;
    tokens = regexp(fileName, pattern, 'tokens');
    if ~isempty(tokens)
        RunIDs = [RunIDs; str2double(tokens{1}{1})]; % Store extracted RunID
    end
end

RunIDs = unique(RunIDs);

selectedRows = ismember(setup.RunID, RunIDs);
setup = setup(~selectedRows, :);


% We will simulate batch by batch. Make sure that the batch size is a multiple
% of the number of subsimulators
% Define the batch size
batchSize = 1;
NumberofBatches = height(setup)/batchSize;

for batch = 1:NumberofBatches
    
    % setup for particular batch
    setup_batch = setup(((batch-1)*batchSize+1):batch*batchSize,:);
 
    % load the simulator
    model = 'MultiLoop_mode1';  
    load_system(model);

    % define the sim objects
    TEObject = strcat(model,'/TE Plant/TE Code');
    distObject0 =strcat(model,"/","Disturbances","/IDV0");
    distObject1 =strcat(model,"/","Disturbances","/IDV1");

    % Create a structure array to keep varying parameter sets
    simIn(1:batchSize) = Simulink.SimulationInput(model);

    % Assign parameters for each simulation input object
    for i = 1:batchSize
        % Assign a random seed and a bit-coded number (i.e., 225) to activate additional
        % measurements
        sfunc_param = strcat('[],',num2str(setup_batch{i,"randomSeed"}),',225');

        % Define the vectors for diturbances
        distA = zeros(1,28);
        distT0 = zeros(1,28);
        distT1 = zeros(1,28);

        % Assign the fault and its amplitude
        distA(setup_batch{i,"fault"}) = setup_batch{i,"amplitude"};
        distT0(setup_batch{i,"fault"}) = setup_batch{i,"startTime"};
        distT1(setup_batch{i,"fault"}) = setup_batch{i,"endTime"};
    
        % Convert those vectors to the suitable format for simulink
        distA = strcat('[',num2str(distA),']');
        distT0 = strcat('[',num2str(distT0),']');
        distT1 = strcat('[',num2str(distT1),']');

        % Finally, set the parameters
        simIn(i) = simIn(i).setBlockParameter(TEObject,'Parameters',sfunc_param);
        simIn(i) = simIn(i).setBlockParameter(distObject0,'after',distA);
        simIn(i) = simIn(i).setBlockParameter(distObject1,'after',distA);
        simIn(i) = simIn(i).setBlockParameter(distObject0,'time',distT0);
        simIn(i) = simIn(i).setBlockParameter(distObject1,'time',distT1);
        simIn(i) = simIn(i).setModelParameter('StopTime', num2str(setup{i,"runTime"}));
    end

    % Run simulations in parallel
    simOut = parsim(simIn, 'ShowProgress', 'on');

    % Find RunIDs to concanate independent simulation results
    RunIDs = unique(setup_batch.RunID);

    for i=1:length(RunIDs)
        indexes = find(setup_batch.RunID == RunIDs(i));
        res = [simOut(1,1).tout,simOut(1,indexes).axmeas, simOut(1,indexes).xmeas, simOut(1,indexes).xmv];
        % Logical mask for columns to keep: 4th row ≠ 5th row
        keep_mask = res(4, :) ~= res(5, :);

        % Apply the mask to retain only non-equal columns
        res = res(:, keep_mask);
        res(:, res(4, :) == res(5, :)) = [];
        amplitude = max(setup_batch.amplitude(indexes));
        rseed =  min(setup_batch.randomSeed(indexes));
        fault =  setup_batch.fault(indexes(1));
        num_of_sims = setup_batch.NumberofSimulators(indexes(1));
        t = setup_batch.runTime(indexes(1));
        t0 = setup_batch.startTime(indexes(1));
        t1 = setup_batch.endTime(indexes(1));

        file_name = strcat(parentDir,"\Data\RunID-",int2str(RunIDs(i)),"-fseed-",int2str(rseed),"-t-",int2str(t),"-ns-",int2str(num_of_sims),"-fault-",int2str(fault), ...
        "-t0-",int2str(t0),"-t1-",int2str(t1),"-amplitude-",num2str(amplitude),"-.csv");
        writematrix(res,file_name)
        writematrix(double(keep_mask), 'keep_mask.csv');
    end
end


