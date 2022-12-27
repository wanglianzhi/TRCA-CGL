    clc;
    clear;
    warning('off', 'all');   
    currentFolder = pwd;           
    addpath(genpath(currentFolder));
    
    dir = '/data';
    dataset = 'ORL_mtv';
    load([dir, '/', dataset]);
    
    % create output dir
    resultDir = ['results/', num2str(dataset), '/'];
    if ~exist(resultDir, 'dir')
       mkdir(resultDir);
    end
    
    
    feaRange = [10:10:100];
    gammas = flip(10 .^ (-1));
    beta        = flip(10 .^ (0));
    delta       = flip(10 .^ (-1));
    lambda   = flip(10 .^ (-1));
    normtds =3;
    nrun = 10;
    weightd = 1;
    
    view_num = length(X); 
    class_num = length(unique(Y)); % number of views, class number
    gnd = Y;
    for i = 1 : view_num
        feaT = [];
        gndT = [];
        for j = 1 : class_num
            indT = find(gnd == j);
            feaT = [feaT, X{i}(indT(:), :)'];
            gndT = [gndT, gnd(indT(:))']; 
        end
        X{i} = feaT;
    end  
    gnd = gndT;
    
    for m = 1 : length(normtds)
          
        normtd = normtds(m); 
        % ===================== Normalization =====================  
        for v = 1 : view_num % transpose the input data matrix after normalization to n * dv;
            if normtd == 1
                X{v} = NormalizeFea(X{v}, 0)';
            elseif normtd == 2
                X{v} = NormalizeData(X{v})';
            else
                X{v} = NormalizeUnit(X{v})';
            end
        end
         for j = 1 : length(gammas)
              for k = 1 : length(beta)
                   for s=1:length(delta)
                       for h = 1:length(lambda)
                    
                  
                        params.gamma = gammas(j);
                        params.beta = beta(k);
                        params.delta = delta(s);
                        params.lambda = lambda(h);
                        params.normtd = normtd;
                        params.nrun = nrun;
                        params.weightd = weightd;
                    
                        MUFS_tensor_OP_wrapper(X, gnd, feaRange, params, resultDir);
                    
                       end
                   end
                end
            end
        end
