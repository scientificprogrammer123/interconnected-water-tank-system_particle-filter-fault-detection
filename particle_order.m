function [X_order,weights_order]=particle_order(Xk,weightsk,nParticles)

[X_order,indexes]=sort(Xk,2);

for i=1:nParticles
    weights_order(1,i)=weightsk(indexes(i));
end