% Comparison of different methods for SMV and MMV cases for
% recovey of common abnormal signal support.
% 
% Methods: TECC (one-step greedy), TECC (SOMP), lasso
% The simulation is done on the followinig setup:
% 1. The signal is N in length and is generated from mixed pdf. 
% 2. The normal pdf is N(0,1)/N(7,1) and abnormal pdf is N(0,10).
% 3. The sensing matrix entries are generated from iid N(0,1).
% 4. The sensing matrix is MxN for each signal.
% 5. J signals are generated for joint recovery.
% 6. K locations of the signal is generated from the abnormal pdf.
% Chenxi Huang 07/28/2017


N=100;% signal length
K=1;% sparsity
indK=N-K+1:N;
mu1=7;
mu2=0;
sig1=1;
sig2=10;
L=5;

Jcol=[2:4,5:5:100];% no. of signals
% Jcol=50;
nJ=length(Jcol);

Mcol=1:10;% no. of measurements
% Mcol=10;
nM=length(Mcol);

niter=500;

accJ_greedy_m=zeros(nJ,nM);
accJ_somp_m=zeros(nJ,nM);
accJ_acie_m=zeros(nJ,nM);
accJ_lasso_m=zeros(nJ,nM);

for m=1:nM
    M=Mcol(m);
    for q=1:nJ
        J=Jcol(q);
        indcol_greedy=zeros(niter,1);
        indcol_somp=zeros(niter,1);
        indcol_acie=zeros(niter,1);
        indcol_lasso=zeros(niter,1);
        for iter=1:niter
            if mod(iter,100)==0
                disp(['M=',num2str(M),',J=',...
                num2str(J),',iter=',...
                num2str(iter)])
            end
            % simulate sensing matrix
            phi=randn(J,M,N);
            % simulate signals
            x=zeros(N,J);
            for j=1:J
               x(1:N-K,j)=mu1*ones(N-K,1)+sig1*randn(N-K,1);
               x(N-K+1:N,j)=mu2*ones(K,1)+sig2*randn(K,1);
            end
            % generate mixed signals
            y=zeros(J,M);
            for j=1:J
               y(j,:)=(reshape(squeeze(phi(j,:,:)),[M,N])*x(:,j))';
            end
            
            % TECC%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % estimate commmon component
            y_comm=zeros(N,1);
            for i=1:N
               for j=1:J
                   y_comm(i)=y_comm(i)+sum(squeeze(phi(j,:,i)).*y(j,:))/M;
               end
            end
            y_comm=y_comm/J;
            % estimate the innovations
            yhat=zeros(size(y));
            for j=1:J
               yhat(j,:)=y(j,:)-(reshape(squeeze(phi(j,:,:)),[M,N])*y_comm)'; 
            end
            % SOMP
            phi_somp=phi/sqrt(N);
            indsel=jsm2_somp(phi_somp,yhat,K);
            indmaxKsort=sort(indsel,'ascend');
            indcol_somp(iter)=sum(abs(indmaxKsort-indK'));
            % one-step greedy
            eps=zeros(N,1);
            for i=1:N
                epscur=0;
                for j=1:J
                    epscur=epscur+sum(yhat(j,:).*squeeze(phi(j,:,i)))^2;
                end
               eps(i)=epscur/J; 
            end
            [~,indmax]=sort(eps,'descend');
            indmaxK=indmax(1:K);
            indmaxKsort=sort(indmaxK,'ascend');
            indcol_greedy(iter)=sum(abs(indmaxKsort-indK'));
            
            % update
            if M>K
                for l=1:L
                    B=phi(:,:,indmax(1:M));
                    Bt=B;
                    for j=1:J
                       [Bt(j,:,:),~]=gschmidt(squeeze(B(j,:,:)));                     
                    end
                    Q=B(:,:,K+1:M);
                    yt=zeros(J,M-K);
                    phit=zeros(J,M-K,N);
                    for j=1:J
                       yt(j,:)=y(j,:)*reshape(squeeze(Q(j,:,:)),[M,M-K]);
                       phit(j,:,:)=reshape(squeeze(Q(j,:,:)),[M,M-K])'*squeeze(phi(j,:,:));
                    end
                    yt_re=reshape(yt',[J*(M-K),1]);
                    phit_re=reshape(permute(phit,[3,2,1]),[N,J*(M-K)])'; 
                    y_comm=pinv(phit_re)*yt_re;
                    % estimate the innovations
%                     yhat=zeros(size(y));
                    for j=1:J
                       yhat(j,:)=y(j,:)-(reshape(squeeze(phi(j,:,:)),[M,N])*y_comm)'; 
                    end
                    % one-step greedy
                    eps=zeros(N,1);
                    for i=1:N
                        epscur=0;
                        for j=1:J
                            epscur=epscur+sum(yhat(j,:).*squeeze(phi(j,:,i)))^2;
                        end
                       eps(i)=epscur/J; 
                    end
                    [~,indmax]=sort(eps,'descend');
                end
            end
            indmaxK=indmax(1:K);
            indmaxKsort=sort(indmaxK,'ascend');
            indcol_acie(iter)=sum(abs(indmaxKsort-indK'));

            % lasso %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % reshape y
            y_re=reshape(y',[J*M,1]);
            % reshape phi
            phi_re=reshape(permute(phi,[3,2,1]),[N,J*M])'; 
            if iter==1 % do CV for lambda selection
                if length(y_re)<3
                    [~,FitInfo]=lasso(phi_re,y_re);
                    [~,minL]=min(FitInfo.MSE);
                    lambda=FitInfo.Lambda(minL);
                elseif length(y_re)<5
                    [~,FitInfo]=lasso(phi_re,y_re,'CV',J);
                    lambda=FitInfo.LambdaMinMSE;
                else
                    [~,FitInfo]=lasso(phi_re,y_re,'CV',5);
                    lambda=FitInfo.LambdaMinMSE;
                end
            end
            [B,~]=lasso(phi_re,y_re,'Lambda',lambda);
            [~,indmax]=sort(B,'descend');
            indmaxK=indmax(1:K);
            indmaxKsort=sort(indmaxK,'ascend');
            indcol_lasso(iter)=sum(abs(indmaxKsort-indK'));
            
        end
        accJ_lasso_m(q,m)=length(find(indcol_lasso==0))/niter;
        accJ_greedy_m(q,m)=length(find(indcol_greedy==0))/niter;
        accJ_somp_m(q,m)=length(find(indcol_somp==0))/niter;
        accJ_acie_m(q,m)=length(find(indcol_mp==0))/niter;
    end
end