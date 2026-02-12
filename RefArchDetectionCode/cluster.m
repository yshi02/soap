% ==============================================================================
%
%                          ===>>>  CLUSTER.M  <<<===
%_______________________________________________________________________________
%
%   PROGRAMMER : Seth Kalson, Jim Ward
%
%   DATE   CODE: 26 August 91
%   UPDATE CODE: 26 August 91
%
%_______________________________________________________________________________
%
%   DESCRIPTION: A function that generates clusters and primitive
%                target reports from the range/doppler detection
%                matrix 'adet' and the matrix of squared
%                magnitudes 'a'. The range/doppler cell with the peak
%                value in each cluster is taken to be the
%                primitive target report.
%
%   The output is a matrix of the same size as 'a' and 'adet'
%_______________________________________________________________________________
%
%   USAGE: atest = cluster(a,adet)
%_______________________________________________________________________________
%
%   INPUTS:  a         : Matrix of range doppler squared magnitudes.
%            adet      : Matrix of detection flags.
%
%   OUTPUTS: atest     : Matrix of clustered target reports.
%_______________________________________________________________________________
%
%   CALLED BY   : pma1cpi
%
%   CALLS TO    : upindex - Updates the index for a cluster.
%
% ==============================================================================

function atest = cluster(a,adet)

x = find(adet) ;
sizex = size(x) ;
sizea = size(adet) ;
index = [1:sizex(1)].' ;

%for i = 1 : sizex(1)-1
for i = 1 : sizex(1)
    if rem(x(i),sizea(1)) ~= 0
       k = find(x==x(i)+1) ;
       index = upindex(k,i,index) ;
    end
    if x(i) <= sizea(1)*(sizea(2)-1) ;
       k = find(x==x(i)+sizea(1)) ;
       index = upindex(k,i,index) ;
       if rem(x(i),sizea(1)) ~= 1
          k = find(x==x(i)+sizea(1)-1) ;
          index = upindex(k,i,index) ;
       end
       if rem(x(i),sizea(1)) ~= 0
          k = find(x==x(i)+sizea(1)+1) ;
          index = upindex(k,i,index) ;
       end
    end
    if x(i) > sizea(1)*(sizea(2)-1)
       k = find(x==x(i)-(sizea(2)-1)*sizea(1)) ;
       index = upindex(k,i,index) ;
       if rem(x(i),sizea(1)) ~= 1
          k = find(x==x(i)-(sizea(2)-1)*sizea(1)-1) ;
          index = upindex(k,i,index) ;
       end
       if rem(x(i),sizea(1)) ~= 0
          k = find(x==x(i)-(sizea(2)-1)*sizea(1)+1) ;
          index = upindex(k,i,index) ;
       end
    end
end

atest = zeros(size(adet)) ;
for i = 1 : sizex(1)
    indexvec = find(index==i) ;
    if indexvec > 0
       xtest = x(indexvec) ;
       [testmax,indexmax] = max(a(xtest)) ;
       atest(xtest(indexmax)) = 1 ;
    end
end

return

