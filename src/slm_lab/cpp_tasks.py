"""C++ contracts/references for the existing hidden local algorithm suite."""

SIGNATURES={
    'sum_even':'Int sum_even(const Ints& xs)',
    'count_runs':'Int count_runs(const Ints& xs)',
    'prefix_sums':'Ints prefix_sums(const Ints& xs)',
    'first_duplicate':'std::optional<Int> first_duplicate(const Ints& xs)',
    'count_target_pairs':'Int count_target_pairs(const Ints& xs, Int target)',
    'lower_bound':'Int lower_bound(const Ints& xs, Int target)',
    'max_window_sum':'std::optional<Int> max_window_sum(const Ints& xs, Int k)',
    'stable_unique':'Ints stable_unique(const Ints& xs)',
    'max_subarray':'Int max_subarray(const Ints& xs)',
    'rotate_left':'Ints rotate_left(const Ints& xs, Int k)',
    'is_balanced':'bool is_balanced(const std::u32string& s)',
    'longest_unique':'Int longest_unique(const std::u32string& s)',
}
RETURNS={'prefix_sums':'vector','stable_unique':'vector','rotate_left':'vector',
         'first_duplicate':'optional','max_window_sum':'optional','is_balanced':'bool'}

REFERENCES={
 'sum_even':'Int out=0;for(Int x:xs)if(x%2==0)out+=x;return out;',
 'count_runs':'Int out=0;for(size_t i=0;i<xs.size();++i)if(i==0||xs[i]!=xs[i-1])++out;return out;',
 'prefix_sums':'Ints out;Int total=0;for(Int x:xs){total+=x;out.push_back(total);}return out;',
 'first_duplicate':'std::unordered_set<Int> seen;for(Int x:xs)if(!seen.insert(x).second)return x;return std::nullopt;',
 'count_target_pairs':'std::unordered_map<Int,Int> seen;Int total=0;for(Int x:xs){total+=seen[target-x];++seen[x];}return total;',
 'lower_bound':'size_t lo=0,hi=xs.size();while(lo<hi){size_t mid=(lo+hi)/2;if(xs[mid]<target)lo=mid+1;else hi=mid;}return static_cast<Int>(lo);',
 'max_window_sum':'if(k<=0||k>static_cast<Int>(xs.size()))return std::nullopt;Int value=std::accumulate(xs.begin(),xs.begin()+k,Int{0}),best=value;for(size_t i=k;i<xs.size();++i){value+=xs[i]-xs[i-k];best=std::max(best,value);}return best;',
 'stable_unique':'std::unordered_set<Int> seen;Ints out;for(Int x:xs)if(seen.insert(x).second)out.push_back(x);return out;',
 'max_subarray':'if(xs.empty())return 0;Int current=xs[0],best=current;for(size_t i=1;i<xs.size();++i){current=std::max(xs[i],current+xs[i]);best=std::max(best,current);}return best;',
 'rotate_left':'if(xs.empty())return {};k=(k%static_cast<Int>(xs.size())+xs.size())%xs.size();Ints out(xs.begin()+k,xs.end());out.insert(out.end(),xs.begin(),xs.begin()+k);return out;',
 'is_balanced':'std::u32string stack;for(char32_t c:s){if(c==U\'(\'||c==U\'[\'||c==U\'{\')stack.push_back(c);else if(c==U\')\'||c==U\']\'||c==U\'}\'){char32_t expected=c==U\')\'?U\'(\':(c==U\']\'?U\'[\':U\'{\');if(stack.empty()||stack.back()!=expected)return false;stack.pop_back();}}return stack.empty();',
 'longest_unique':'std::unordered_map<char32_t,Int> last;Int start=0,best=0;for(size_t i=0;i<s.size();++i){auto found=last.find(s[i]);if(found!=last.end())start=std::max(start,found->second+1);last[s[i]]=i;best=std::max(best,static_cast<Int>(i)-start+1);}return best;',
}

HEADERS='''#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
using Int=long long;
using Ints=std::vector<Int>;
'''


def reference(name):return SIGNATURES[name]+' { '+REFERENCES[name]+' }\n'


def harness(name):
    argument='s' if name in {'is_balanced','longest_unique'} else 'xs'
    if name in {'count_target_pairs','lower_bound','max_window_sum','rotate_left'}:argument+=',extra'
    category=RETURNS.get(name,'int')
    output={'vector':'std::cout<<answer.size();for(Int value:answer)std::cout<<" "<<value;',
            'optional':'if(answer)std::cout<<*answer;else std::cout<<"null";',
            'bool':'std::cout<<(answer?1:0);','int':'std::cout<<answer;'}[category]
    return '''#include "contract.hpp"
#include <iostream>
#include <sstream>
int main(){
std::string line;
while(std::getline(std::cin,line)){
std::istringstream input(line);size_t count=0,repetitions=1;Int extra=0;
if(!(input>>count)||count>1000000)return 4;
Ints xs;xs.reserve(count);std::u32string s;
for(size_t i=0;i<count;++i){Int value;if(!(input>>value))return 4;xs.push_back(value);s.push_back(static_cast<char32_t>(value));}
if(!(input>>extra>>repetitions)||repetitions>10000)return 4;
try{
auto answer='''+name+'('+argument+''');
for(size_t i=1;i<repetitions;++i)answer='''+name+'('+argument+''');
'''+output+'''
std::cout<<std::endl;
}catch(const std::exception& error){std::cerr<<error.what()<<std::endl;return 3;}
}
return 0;
}
'''
