function [score] = TRCA_CGL(X, X_fea,view_num, N, d, class_num, params)
    gamma = params.gamma;
    beta      = params.beta;
    delta     = params.delta;
    lambda = params.lambda;
    weightd = params.weightd;
    alphaV = ones(1, view_num) / view_num; 
    rho = 1e-2; pho_rho =2; max_rho = 1e12; eta=1e8; 
    ratio=1;
    for v=1:view_num
        sigma(v)=ratio*optSigma(X{v});
    end
    %% Construct kernel and transition matrix P{v}
    K=[];
    P=cell(1,view_num);
    for j=1:view_num
        options.KernelType = 'Gaussian';
        options.t = sigma(j);
        K(:,:,j) = constructKernel(X{j},X{j},options);
        D=diag(sum(K(:,:,j),2));
        L_rw=D^-(1/2)*K(:,:,j)*D^-(1/2);
        P{j}=L_rw;
    end
    %% initial 
    E = cell(view_num, 1);
    Z = cell(view_num, 1);
    G = cell(view_num, 1);
    R = cell(view_num, 1);
    Q = cell(view_num, 1);
    W =rand(d, class_num);
    M=constructW_PKN(X_fea',9);
    tempM=(M+M')/2;
    Dm=diag(sum(tempM));
    Lm=Dm-M;
    [F,~,~]=eig1(Lm,class_num,0);
    for v = 1 : view_num
        E{v}= zeros(N,N);
        Z{v}= zeros(N,N);
    end
%% initialize auxiliary variable G,lagrange multiplier R
    for v = 1 : view_num
        R{v} = zeros(N,N);
        G{v} = zeros(N,N);
        Q{v} = zeros(N,N);
    end
%% optimization
    iter = 1; max_iter = 200;
    error = zeros(50, 1);
    while iter <= max_iter       
        fprintf('----processing iter %d--------\n', iter);
        %% optimization tensor
        Z_tensor = cat(3, Z{:,:});
        R_tensor = cat(3, R{:,:});
        Zv = Z_tensor(:);
        Rv = R_tensor(:);
        sT = [N, N, view_num];
        [Gv, ~] = wshrinkObj(Zv + 1 / rho * Rv,N*1/ rho, sT, 0, 3);
        G_tensor = reshape(Gv, sT);
        for v = 1 : view_num
            G{v} = G_tensor(:, :, v);
        end
         %% optimization Z{v}
        for v=1:view_num
            A{v}=P{v}-E{v}+(1/rho)*Q{v};
            B{v}=G{v}-(1/rho)*R{v};
            Z{v}=(2*alphaV(v)*M+rho*(A{v}+B{v}))/(2*(alphaV(v)+rho));
        end
        %% optimization E{v}
        T=[P{1}-Z{1}+Q{1}/rho;P{2}-Z{2}+Q{2}/rho;P{3}-Z{3}+Q{3}/rho];
        [Econcat]=solve_l1l2(T,gamma/rho);
        E{1}=Econcat(1:size(P{1},1),:);
        E{2}=Econcat(size(P{1},1)+1:size(P{1},1)+size(P{2},1),:);
        E{3}=Econcat(size(P{1},1)+size(P{2},1)+1:end,:);
        %% optimization M
        dist = L2_distance_1(F',F');
        M = zeros(N);  
        for i=1:N   
            a0 = zeros(1,N);
            for v = 1:view_num
                temp = Z{v};
                a0 = a0+alphaV(1,v)*temp(i,:);
            end
            idxa0 = find(a0>0);
            ai = a0(idxa0);
            di = dist(i,idxa0);
            ad = (ai-0.5*lambda*di)/sum(alphaV);  
            M(i,idxa0) = EProjSimplex_new(ad);
        end
      
         %% optimization W
        xwf=X_fea*W-F;
        temp_dd = zeros(d,1);temp_hh=zeros(N,1); 
        for j=1:d
            temp_dd(j)=1/(2*sqrt(W(j,:)*W(j,:)')+eps);
        end
        for i=1:N
            temp_hh(i)=1/(2*sqrt(xwf(i,:)*xwf(i,:)')+eps);
        end
        DD=diag(temp_dd);HH=diag(temp_hh);
        W=(X_fea'*HH*X_fea+delta*DD)\X_fea'*HH*F;
            
        %% optimization F
        tempM=(M+M')/2;
        Dm=diag(sum(tempM));
        Lm=Dm-M;
        temp_jj = zeros(N,1);
        XWF=X_fea*W-F;
        for j=1:N
            temp_jj(j)=1/(2*sqrt(XWF(j,:)*XWF(j,:)')+eps);
        end
        JJ=diag(temp_jj);
        Ftop=beta*JJ*X_fea*W+eta*F;
        Fdown=beta*JJ*F+Lm*F+eta*F*(F'*F)+eps;
        Frac=Ftop./Fdown;
        F=F.*Frac;
        F=F*diag(sqrt(1./(diag(F'*F)+eps)));     
%% update view weights
        if weightd == 1
            for v = 1 : view_num
                alphaV(v) = 0.5/norm(M-Z{v},'fro');      
            end
        end  
        Rv = Rv + rho * (Zv - Gv);
        R_tensor = reshape(Rv, sT);
        for v=1:view_num
            Q{v}=Q{v}+rho *(P{v}-Z{v}-E{v});
            R{v} = R_tensor(:, :, v);
        end   
%% check convergence
    leqm1=0;
    for v = 1:view_num
        Rec_error = P{v}-Z{v}-E{v};
        leqm1 = max(leqm1,max(abs(Rec_error(:))));
    end
    leq = Z_tensor-G_tensor; 
    leqm = max(abs(leq(:)));
    err = max([leqm1,leqm]);
    fprintf('iter = %d, rou = %.3f, leqm1 = %.8f, leqm2 = %.8f,err = %d\n', iter, rho, leqm1,leqm,err);
    error(iter) = err;
     if err < 1e-6
         break;
     end
     iter = iter + 1;
     rho = min(rho * pho_rho, max_rho);         
    end   
    score = sqrt(sum(W .* W, 2));
end


