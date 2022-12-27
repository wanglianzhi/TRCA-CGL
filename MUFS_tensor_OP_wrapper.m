function MUFS_tensor_OP_wrapper(X, gnd, feaRange, params, resultDir)

    view_num = length(X);
    N = size(X{1}, 1);              
    class_num = length(unique(gnd)); 
    nrun = params.nrun;
    X_fea = DataConcatenate(X);
    d=size(X_fea,2);
    
    ACC_fid = fopen([resultDir,  '_gamma_', num2str(params.gamma), '_normtd_', num2str(params.normtd) ...
                                   '_beta_', num2str(params.beta),'_delta_', num2str(params.delta),'_lambda_', num2str(params.lambda),'_weightd_', num2str(params.weightd), '_ACC_', '.txt'], 'w'); 
    NMI_fid = fopen([resultDir,  '_gamma_', num2str(params.gamma), '_normtd_', num2str(params.normtd) ...
                                    '_beta_', num2str(params.beta),'_delta_', num2str(params.delta),'_lambda_', num2str(params.lambda), '_weightd_', num2str(params.weightd), '_NMI_', '.txt'], 'w'); 
    Purity_fid = fopen([resultDir, '_gamma_', num2str(params.gamma), '_normtd_', num2str(params.normtd) ...
                                   '_beta_', num2str(params.beta),'_delta_', num2str(params.delta),'_lambda_', num2str(params.lambda),'_weightd_', num2str(params.weightd), '_Purity_', '.txt'], 'w');                            
    P_fid        = fopen([resultDir,  '_gamma_', num2str(params.gamma), '_normtd_', num2str(params.normtd) ...
                                   '_beta_', num2str(params.beta),'_delta_', num2str(params.delta), '_lambda_', num2str(params.lambda),'_weightd_', num2str(params.weightd), '_P_', '.txt'], 'w'); 
    R_fid        = fopen([resultDir,  '_gamma_', num2str(params.gamma), '_normtd_', num2str(params.normtd) ...
                                   '_beta_', num2str(params.beta),'_delta_', num2str(params.delta),'_lambda_', num2str(params.lambda), '_weightd_', num2str(params.weightd), '_R_', '.txt'], 'w');    
    F_fid        = fopen([resultDir, '_gamma_', num2str(params.gamma), '_normtd_', num2str(params.normtd) ...
                                   '_beta_', num2str(params.beta),'_delta_', num2str(params.delta), '_lambda_', num2str(params.lambda),'_weightd_', num2str(params.weightd), '_F_', '.txt'], 'w');
    RI_fid        = fopen([resultDir, '_gamma_', num2str(params.gamma), '_normtd_', num2str(params.normtd) ...
                                   '_beta_', num2str(params.beta),'_delta_', num2str(params.delta), '_lambda_', num2str(params.lambda),'_weightd_', num2str(params.weightd), '_RI_', '.txt'], 'w');
    header = strjoin(cellstr(strsplit(num2str(feaRange))), '\t');
    header = ['nrun\t', header, '\n'];

    fprintf(ACC_fid, header);
    fprintf(NMI_fid, header);
    fprintf(Purity_fid, header);
    fprintf(P_fid, header);
    fprintf(R_fid, header);
    fprintf(F_fid, header);
    fprintf(RI_fid, header);
    ACC = zeros(nrun, length(feaRange));
    NMI = zeros(nrun, length(feaRange));
    purity = zeros(nrun, length(feaRange));
    P = zeros(nrun, length(feaRange));
    R = zeros(nrun, length(feaRange));
    F = zeros(nrun, length(feaRange));
    RI = zeros(nrun, length(feaRange));
    
    for v = 1 : nrun
            rng(v, 'twister'); 
            [score] = TRCA_CGL(X, X_fea,view_num, N, d, class_num, params);
            [~, index] = sort(score,'descend');
            for j = 1 : length(feaRange)
                selecteddata = X_fea(:, index(1 : feaRange(j)));
                [label_predict, ~] = litekmeans(selecteddata, class_num);
                result = ClusteringMeasure1(gnd, label_predict);
                ACC(v, j) = result(1);
                NMI(v, j) = result(2);
                purity(v, j) = result(3);
                P(v, j) =result(4);
                R(v, j) = result(5);
                F(v, j) = result(6);
                RI(v, j) =result(7);
            end

            fprintf(ACC_fid, [strjoin(cellstr(strsplit(num2str([v, ACC(v,:)]))), '\t'), '\n']);
            fprintf(NMI_fid, [strjoin(cellstr(strsplit(num2str([v, NMI(v,:)]))), '\t'), '\n']);
            fprintf(Purity_fid, [strjoin(cellstr(strsplit(num2str([v, purity(v,:)]))), '\t'), '\n']);
            fprintf(P_fid, [strjoin(cellstr(strsplit(num2str([v, P(v,:)]))), '\t'), '\n']);
            fprintf(R_fid, [strjoin(cellstr(strsplit(num2str([v, R(v,:)]))), '\t'), '\n']);
            fprintf(F_fid, [strjoin(cellstr(strsplit(num2str([v, F(v,:)]))), '\t'), '\n']);
            fprintf(RI_fid, [strjoin(cellstr(strsplit(num2str([v,RI(v,:)]))), '\t'), '\n']);
    end
    fclose(ACC_fid);
    fclose(NMI_fid);
    fclose(Purity_fid);
    fclose(P_fid);
    fclose(R_fid);
    fclose(F_fid);
    fclose(RI_fid);
end

