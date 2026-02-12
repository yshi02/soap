%
%  upindex.m
%  script file for updating cluster index vector.  Called by cluster
%  global variables are k,i, and index vector
%
function idxdum=upindex(k,i,index)
if k > 0
   if index(i) <= index(k)
      index(k)=index(i);
   else
      vectemp=find(index==index(i));
      tempsize=size(vectemp);
      index(vectemp)=index(k)*ones(size(1:tempsize(1)));
   end
end
idxdum=index;
return
