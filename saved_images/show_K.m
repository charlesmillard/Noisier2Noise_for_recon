clear all

p_omega = 0:0.001:1;
p_lambda = 0:0.001:1;

k = zeros(numel(p_omega), numel(p_lambda));

for ii=1:numel(p_omega)
    k(end - ii + 1,:) = (1-p_omega(ii))./(1 - p_omega(ii).*p_lambda);
end

imshow(k)
colormap("parula")
colorbar;
xlabel('P lambda')
ylabel('P omega')