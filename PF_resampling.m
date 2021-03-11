    %% [Xk_1_1_pf,weightsk_1_1_pf]=PF_resampling(Xk_1_pf,weightsk_1_pf,M,Thr)
function [Xk_1, weightsk_1] = PF_resampling(Xk, weightsk, nParticles, Thr, k)
%Xk consists of:
%Xk_1_pf               system estimate for tank 1
%Xk_2_pf               system estimate for tank 2
%Xk_3_pf               system estimate for tank 3
%X_boolean_t2_pf       for leakage in tank2 system
%weightsk              these are the weights before resampling
%nParticles            10
%thr                   0.65 
%k                     212

%Generate 1((weights sum)^2), compare it to threshold (0.65*10=6.5).
%
%If the reciprocal of the square of the sum of weights(k) is higher than 
%the threshold, then the particles X(k-1) are updated to X(k), and the weights
%weights(k-1) are also updates to weights(k).
% 
%A high (sum(weights))^2 means the entire set ofprior belief is close to the 
%measurement. Inversely, a low (sum(weights))^2 means the entire set of prior 
%belief is far away from the measurement, or is a poor representation of the measurement.
%
%If the entire set of prior particles are too far away from the
%measurement, throw out the entire set of weights and update with new
%weights, else keep the entire set of previous weights.
%
%Do this before resampling step.
    wSum_Sq = weightsk * weightsk'; 
    if ((1/wSum_Sq) > (Thr*nParticles)) %these 2 statements should be equivalent
    %if ((wSum_Sq) < ((Thr*nParticles)^(-1))) 
        %disp('old set of particles suck, use new particles');
        Xk_1 = Xk;                       
        weightsk_1 = weightsk;    
        %fprintf('weights%i less than threshold, use Xk_1 and weightsk_1\n', k);
    end;

%Resample Step:
%
%The real �trick� of the particle filter algorithm occurs in Lines 8 through 11 in Table
%4.3. These lines implemented what is known as resampling or importance resampling.
%The algorithm draws with replacement M particles from the temporary
%set(X_t)^(~).
%
%The probability of drawing each particle is given by its importance weight.
%Resampling transforms a particle set of M particles into another particle set of the
%same size. By incorporating the importance weights into the resampling process,
%the distribution of the particles change: whereas before the resampling step, they
%were distribution according to (bel(x_t))^(~), after the resampling they are distributed
%(approximately) according to the posterior bel(x_t) = nu p(z_t | (x_t)^[m]) (bel(x_t))^(~). 
%
%In fact, the resulting sample set usually possesses many duplicates, since particles
%are drawn with replacement. More important are the particles that are not contained
%in X_t: those tend to be the particles with lower importance weights.
%
%How is this implemented below?

    % generate weights CDF and plot it below
    % u goes from 0 to 1
    % c goes from 0 to 1
    c = zeros(nParticles, 1);           %x
    u = c;                              %y
    u(1) = 0.5/nParticles;              % = 0.5/100 = 0.005
    c(1) = weightsk(1,1);               % = weight of the first particle in the current time instance

%build CDF
    for i = 2:nParticles
        c(i) = c(i-1) + weightsk(i);    %y=particle weight CDF
        u(i) = u(i-1) + 1/nParticles;   %x=bins 
    end;

    %{
    %plot CDF
    figure(15);
    %subplot(2,1,1);
    plot(u,c,'-o');
    grid on;
    grid minor;
    %axis([0 100 0 1]);
    axis([0 1 0 1]);
    %xlabel('particle'), ylabel('weight CDF'); 
    xlabel('uniform weights CDF, u'), ylabel('weight(k) CDF, c'); 
    title('uniform weights CDF vs. weights(k) CDF');
    style = hgexport('factorystyle');
    style.Color = 'gray';
    %title('unifom weights CDF');
    %subplot(2,1,2);
    %plot(c,'-o');
    %grid on;
    %grid minor;
    %axis([0 100 0 1]);
    %xlabel('particle'), ylabel('weight CDF'); 
    %title('weights(k) CDF');
    %}

    i = 1;

%For each particle in the list of particles (M, or nParticles), check to
%see if its weight is higher than that of a particle in a uniform distribtion, 
%this is the criterion for replacement.
%If its weight is higher, then insert this particle into X(k-1), which will 
%be used in the next iteration of the estimation.
    for j = 1:nParticles
    
        %Does this particle in the M list of weights(k) prior to resampling have 
        %a higher importance than a particle from a uniform distribution? 
        %If yes, note its index.
        while u(j) >= c(i)
            i = i + 1;
        end;
        %i represent the index of the particle whose weight CDF is higher than
        %the weight CDF of a uniform distribution up to that point.
        
        %replace the jth particle in X(k-1) set, consisting of (t1, t2, t3, boolean) 
        %with the ith particle in X(k), ok
        Xk_1(:,j) = Xk(:,i);
    
        %set the weight of this replaced particle to 1/N, so the default weight 
        %assigned to a particle with replacement is particle weight of uniform
        %distribution, ok
        weightsk_1(1,j) = 1/nParticles; 
        
        %{
        fileID = fopen('exp.txt','a');
        %formatSpec = ;
        fprintf(fileID, 'main i index is, %d, j index in pf resampling is, %d, i index in pf resampling is, %d\n', k, j, i);
        %formatSpec = ;
        %fprintf(fileID, 'i index in pf resampling is %d\n', i)
        fclose(fileID);
        %}
    
    end;

%{
if k<100
    pause
end
%}

%why is it that it's always the 99th or 100th particle in the set that has
%a higher importance than the uniform distribution? WTF?

%From Udacity:
%An obvious way to do this might be to compute all these normalized alphas,
%So in the spectrum of our alphas (normalized weights) you might draw a random 
%variable uniformly from the interval [0:1] called beta, and then find out 
%the alpha such that all the alphas leading up to it, and some are smaller 
%than beta, but if we add the new alpha to the sum you would get a value 
%larger than beta. Beta is the CDF not the PDF of uniform random variable.

%There is another implementation of resampling in Udacity course CS373 AI for
%robotics, it uses a resampling wheel as a trick.

