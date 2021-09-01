#include<bits/stdc++.h>
using namespace std;

int fa[10005];
map<string,int>si;
map<int,string>is;
vector<string>ans;
string s[10005][3];
bool dfs(int a, int b)
{
    if (a==0)
        return false;
    else if (a==b)
        return true;
    else
        return dfs(fa[a],b);
}
int main()
{
    int n;
    cin>>n;
    int js =1;
    memset(fa,0,sizeof(fa));
    for(int i=0;i<n;i++)
    {
        cin>>s[i][0]>>s[i][1]>>s[i][2];
        if (si[s[i][0]]==0)
        {
            si[s[i][0]]=js;
            is[js]=s[i][0];
            js++;
        }
        if (si[s[i][2]]==0)
        {
            si[s[i][2]]=js;
            is[js]=s[i][2];
            js++;
        }
        //cout<<s[i][0]<<" "<<s[i][1]<<" "<<s[i][2]<<endl;
        fa[si[s[i][0]]]=si[s[i][2]];
        
    }
    string q;
    cin>>q;
    //cout<<q<<endl;
    for (int i=0;i<n;i++)
    {
        if (s[i][1]=="instanceOf" && dfs(fa[si[s[i][0]]],si[q])==true)
            {
                ans.push_back(is[si[s[i][0]]]);
            }
    }
    if (ans.size()==0)
    {
        cout<<"empty"<<endl;
    }
    else{
    sort(ans.begin(),ans.end());
    for (int i=0;i<ans.size();i++)
    {
        if (i!=ans.size()-1)
            cout<<ans[i]<<" ";
        else
            cout<<ans[i]<<endl;
    }
    }
}
