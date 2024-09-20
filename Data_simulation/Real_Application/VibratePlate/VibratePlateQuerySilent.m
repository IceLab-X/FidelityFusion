function y = VibratePlateQuerySilent(X,m)

%if m == 0
%    hmax=1.2;
%elseif m == 1
%    hmax=0.6;
%else
%   fprintf('ERROR: Fidelity not defined\n')
%end

hmax = 1.2-0.6*m;

Ns = size(X,1);
y = zeros(Ns,1);

for i = 1:Ns
    s = X(i, :);
    E = s(1);
    nu = s(2);
    mu = s(3);
    
    yi = VibrationPlate(E,nu,mu,hmax);
    y(i) = yi;
    
end

% fprintf('$');
% for i = 1:Ns
% 
%     fprintf('%.8f', y(i));
%     if i < Ns
%         fprintf(',')
%     end
% 
% end
% fprintf('$')


end