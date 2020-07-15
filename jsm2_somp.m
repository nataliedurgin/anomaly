function indsel=jsm2_somp(phi,y,K)
[J,M,N]=size(phi);
gamma=zeros(J,M,K);
resid=y;
indsel=zeros(K,1);% index set
for k=1:K
                % greedy selection
                eps=zeros(N,1);
                for i=1:N
                    epscur=0;
                    for j=1:J
                        epscur=epscur+abs(sum(resid(j,:).*squeeze(phi(j,:,i))));
                    end
                    eps(i)=epscur; 
                end
                [~,ind]=max(eps);
                indsel(k)=ind;
                % orthogonalization
                if k==1
                    gamma(:,:,k)=phi(:,:,ind);
                else
                    for j=1:J
                        for t=1:k-1
                           gamma(j,:,k)=phi(j,:,ind)...
                               -(sum(squeeze(phi(j,:,ind)).*squeeze(gamma(j,:,t)))...
                               /sum(squeeze(gamma(j,:,t)).^2))*gamma(j,:,t); 
                        end
                    end
                end
                % update residual
                for j=1:J
                    resid(j,:)=resid(j,:)...
                        -(sum(squeeze(resid(j,:)).*squeeze(gamma(j,:,k)))...
                        /sum(squeeze(gamma(j,:,k)).^2))*squeeze(gamma(j,:,k));
                end
end