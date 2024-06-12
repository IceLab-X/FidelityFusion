function y = HeatedBlockQuery(X,m)


if m == 0
    Hmax=0.8;
elseif m == 1
    Hmax=0.2;
else
    fprintf('ERROR: Fidelity not defined\n')
end

Ns = size(X,1);
y = zeros(Ns,1);

for i = 1:Ns
    s = X(i, :);
    lx = s(1);
    ly = s(2);
    theta = s(3);
    
    try
        yi = heatedBlock2(lx,ly,theta, Hmax);
        y(i) = yi;
    catch
        y(i) = heatedBlock2(0.4,0.4,0, Hmax);
    end
    
end

fprintf('$');
for i = 1:Ns

    fprintf('%.8f', y(i));
    if i < Ns
        fprintf(',')
    end

end
fprintf('$')


end