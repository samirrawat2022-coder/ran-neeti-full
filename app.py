import io, json, zipfile, requests, re
from datetime import datetime
from collections import defaultdict, deque
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, f1_score, brier_score_loss

st.set_page_config(page_title='RAN-NEETI | AI Prediction Lab', page_icon='◈', layout='wide')

st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
:root{--bg:#070b14;--panel:#0e1626;--line:#24344d;--cyan:#53e5ff;--violet:#a78bfa;--muted:#91a2bd;}
.stApp{background:radial-gradient(ellipse at 12% 0%,#15233c 0%,#070b14 45%,#080b16 100%);color:#edf4ff;font-family:Inter,sans-serif}
.block-container{padding-top:1.4rem;max-width:1250px}
.brand{font-family:Orbitron,sans-serif;font-size:2.35rem;letter-spacing:.14em;font-weight:800;color:#f4f8ff;text-shadow:0 0 22px #53e5ff45}
.brand span{color:var(--cyan)} .sub{color:#91a2bd;letter-spacing:.18em;text-transform:uppercase;font-size:.72rem}
.panel{background:linear-gradient(145deg,#101b2d,#0b1220);border:1px solid #263954;border-radius:16px;padding:20px 22px;margin:8px 0;box-shadow:0 8px 35px #0002}
.kicker{font-family:Orbitron,sans-serif;color:#53e5ff;font-size:.72rem;letter-spacing:.15em;text-transform:uppercase}
.big{font-family:Orbitron,sans-serif;font-size:1.7rem;font-weight:700;color:#fff}
.muted{color:#91a2bd;font-size:.88rem}.tag{display:inline-block;border:1px solid #2a4763;border-radius:999px;padding:4px 10px;color:#8deeff;font-size:.72rem;margin:3px}
hr{border-color:#23344c!important} h1,h2,h3{font-family:Orbitron,sans-serif!important}
.stButton>button{background:linear-gradient(90deg,#23c9e8,#8b7cff);color:#06101a;border:0;border-radius:10px;font-weight:800;min-height:2.8rem}
.stButton>button:hover{filter:brightness(1.1);color:#06101a;border:0}
[data-testid="stMetric"]{background:#0d1727;border:1px solid #263954;padding:14px;border-radius:12px}
[data-testid="stSidebar"]{background:#080e19;border-right:1px solid #263954}
</style>''', unsafe_allow_html=True)

FOOTBALL_SEASONS = ['2324','2223','2122','2021','1920','1819']
FOOTBALL_URL = 'https://www.football-data.co.uk/mmz4281/{season}/E0.csv'
CRICKET_URL = 'https://cricsheet.org/downloads/t20s_json.zip'
WIKI_API = 'https://en.wikipedia.org/w/api.php'

@st.cache_data(ttl=3600, show_spinner=False)
def load_football():
    frames=[]; notes=[]
    for season in FOOTBALL_SEASONS:
        try:
            r=requests.get(FOOTBALL_URL.format(season=season),timeout=18,headers={'User-Agent':'RAN-NEETI academic project/1.0'})
            r.raise_for_status(); df=pd.read_csv(io.BytesIO(r.content), encoding='latin1', on_bad_lines='skip')
            required={'HomeTeam','AwayTeam','FTHG','FTAG','FTR'}
            if required.issubset(df.columns):
                df=df[list(required)+(['Date'] if 'Date' in df.columns else [])].copy(); df['Season']=season; frames.append(df)
        except Exception as e: notes.append(f'{season}: {type(e).__name__}')
    if not frames: return pd.DataFrame(), notes
    out=pd.concat(frames,ignore_index=True)
    if 'Date' in out:
        out['ParsedDate']=pd.to_datetime(out['Date'],dayfirst=True,errors='coerce')
    else: out['ParsedDate']=pd.NaT
    out=out.dropna(subset=['HomeTeam','AwayTeam','FTHG','FTAG','FTR'])
    out=out[out['FTR'].isin(['H','D','A'])].copy()
    out['FTHG']=pd.to_numeric(out.FTHG,errors='coerce'); out['FTAG']=pd.to_numeric(out.FTAG,errors='coerce')
    out=out.dropna(subset=['FTHG','FTAG'])
    out=out.sort_values('ParsedDate',na_position='last').reset_index(drop=True)
    return out, notes

@st.cache_data(ttl=86400, show_spinner=False)
def load_cricket():
    r=requests.get(CRICKET_URL,timeout=45,headers={'User-Agent':'RAN-NEETI academic project/1.0'}); r.raise_for_status()
    z=zipfile.ZipFile(io.BytesIO(r.content)); rows=[]
    files=[n for n in z.namelist() if n.endswith('.json')]
    for name in files:
        try:
            obj=json.loads(z.read(name)); info=obj.get('info',{}); teams=info.get('teams',[]); outcome=info.get('outcome',{})
            winner=outcome.get('winner'); dates=info.get('dates',[])
            if len(teams)==2 and winner in teams and dates:
                rows.append({'date':str(dates[0]),'team1':teams[0],'team2':teams[1],'winner':winner,'venue':info.get('venue','Unknown')})
        except Exception: continue
    df=pd.DataFrame(rows)
    if not df.empty: df['date']=pd.to_datetime(df.date,errors='coerce'); df=df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def wiki_search(query):
    params={'action':'query','list':'search','srsearch':query,'format':'json','srlimit':5}
    r=requests.get(WIKI_API,params=params,timeout=15,headers={'User-Agent':'RAN-NEETI academic project/1.0'}); r.raise_for_status()
    return r.json().get('query',{}).get('search',[])

@st.cache_data(ttl=3600, show_spinner=False)
def wiki_summary(title):
    params={'action':'query','prop':'extracts|info','exintro':1,'explaintext':1,'inprop':'url','redirects':1,'titles':title,'format':'json'}
    r=requests.get(WIKI_API,params=params,timeout=15,headers={'User-Agent':'RAN-NEETI academic project/1.0'}); r.raise_for_status()
    pages=r.json().get('query',{}).get('pages',{})
    if not pages: return None
    p=next(iter(pages.values())); return {'title':p.get('title',title),'extract':p.get('extract',''),'url':p.get('fullurl','https://en.wikipedia.org/wiki/'+title.replace(' ','_'))}

def recent_features(df, home, away, before_idx=None, cricket=False):
    if cricket:
        history=defaultdict(lambda: deque(maxlen=5))
        upto=len(df) if before_idx is None else before_idx
        for _,r in df.iloc[:upto].iterrows():
            history[r.team1].append(1 if r.winner==r.team1 else 0)
            history[r.team2].append(1 if r.winner==r.team2 else 0)
        h=np.mean(history[home]) if history[home] else .5; a=np.mean(history[away]) if history[away] else .5
        return {'team1':home,'team2':away,'form_diff':h-a,'form_team1':h,'form_team2':a}
    history=defaultdict(lambda: deque(maxlen=5)); gd=defaultdict(lambda: deque(maxlen=5))
    upto=len(df) if before_idx is None else before_idx
    for _,r in df.iloc[:upto].iterrows():
        h,a=r.HomeTeam,r.AwayTeam; hg,ag=float(r.FTHG),float(r.FTAG)
        history[h].append(1 if hg>ag else 0.5 if hg==ag else 0); history[a].append(1 if ag>hg else 0.5 if ag==hg else 0)
        gd[h].append(hg-ag); gd[a].append(ag-hg)
    return {'HomeTeam':home,'AwayTeam':away,'home_form':np.mean(history[home]) if history[home] else .5,'away_form':np.mean(history[away]) if history[away] else .5,'home_gd':np.mean(gd[home]) if gd[home] else 0,'away_gd':np.mean(gd[away]) if gd[away] else 0}

def train_football(df):
    feats=[]; targets=[]
    for i,r in df.iterrows():
        f=recent_features(df,r.HomeTeam,r.AwayTeam,before_idx=i); feats.append(f); targets.append(r.FTR)
    X=pd.DataFrame(feats); y=np.array(targets)
    cut=max(int(len(X)*.8),1)
    cat=['HomeTeam','AwayTeam']; num=['home_form','away_form','home_gd','away_gd']
    pipe=Pipeline([('prep',ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),cat),('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num)])),('model',LogisticRegression(max_iter=1200,class_weight='balanced'))])
    if len(X[:cut])<30 or len(set(y[:cut]))<2: raise ValueError('Not enough football data to train a stable baseline.')
    pipe.fit(X.iloc[:cut],y[:cut]); test=X.iloc[cut:]; yt=y[cut:]
    metrics={}
    if len(test):
        pred=pipe.predict(test); proba=pipe.predict_proba(test)
        metrics={'Accuracy':accuracy_score(yt,pred),'Macro F1':f1_score(yt,pred,average='macro',zero_division=0),'Log loss':log_loss(yt,proba,labels=pipe.named_steps['model'].classes_),'Test matches':len(test),'Train matches':cut}
    # Retrain the deployment model on all currently available historical rows after keeping the temporal holdout metrics.
    pipe.fit(X,y)
    return pipe,metrics

def train_cricket(df):
    feats=[]; targets=[]
    for i,r in df.iterrows():
        f=recent_features(df,r.team1,r.team2,before_idx=i,cricket=True); feats.append(f); targets.append(1 if r.winner==r.team1 else 0)
    X=pd.DataFrame(feats); y=np.array(targets); cut=max(int(len(X)*.8),1)
    cat=['team1','team2']; num=['form_diff','form_team1','form_team2']
    pipe=Pipeline([('prep',ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),cat),('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num)])),('model',LogisticRegression(max_iter=1200,class_weight='balanced'))])
    if len(X[:cut])<30 or len(set(y[:cut]))<2: raise ValueError('Not enough cricket data to train a stable baseline.')
    pipe.fit(X.iloc[:cut],y[:cut]); test=X.iloc[cut:]; yt=y[cut:]; metrics={}
    if len(test):
        pred=pipe.predict(test); prob=pipe.predict_proba(test)[:,1]
        metrics={'Accuracy':accuracy_score(yt,pred),'Macro F1':f1_score(yt,pred,average='macro',zero_division=0),'Brier score':brier_score_loss(yt,prob),'Test matches':len(test),'Train matches':cut}
    pipe.fit(X,y); return pipe,metrics

# Header
st.markdown('<div class="brand">RAN<span>·</span>NEETI</div><div class="sub">Unified AI Prediction Lab · Research prototype</div>',unsafe_allow_html=True)
st.write('')
with st.sidebar:
    st.markdown('### ◈ CONTROL DECK')
    domain=st.radio('Prediction domain',['Real Sports','Anime / Fiction'],horizontal=False)
    mode=st.radio('Input mode',['Automatic','Advanced controls'],index=0)
    st.divider()
    st.caption('Free-first build · no API key required')
    st.caption('Sports data is fetched from public historical sources. Fictional probabilities are subjective estimates, not validated ML outputs.')

if domain=='Real Sports':
    sport=st.selectbox('SPORT ENGINE',['Football / Soccer','Cricket · T20'])
    if sport=='Football / Soccer':
        st.markdown('<div class="panel"><div class="kicker">01 / Match intelligence</div><h2>Football outcome prediction</h2><div class="muted">Historical results → recent form features → Logistic Regression → temporal holdout evaluation.</div></div>',unsafe_allow_html=True)
        col1,col2=st.columns(2)
        with col1: home=st.text_input('HOME TEAM','Arsenal')
        with col2: away=st.text_input('AWAY TEAM','Liverpool')
        advanced=False
        if mode=='Advanced controls':
            advanced=True
            seasons=st.multiselect('Historical seasons to use',FOOTBALL_SEASONS,default=FOOTBALL_SEASONS[:4],help='Season codes used by Football-Data.co.uk. Current default uses recent seasons.')
            st.caption('Home advantage is represented by home/away team roles. The baseline uses only historical results available before each match.')
        else: seasons=FOOTBALL_SEASONS
        if st.button('⚡ RUN SPORTS PREDICTION',use_container_width=True):
            with st.spinner('Retrieving historical results and training baseline…'):
                try:
                    df,notes=load_football()
                    if advanced and seasons: df=df[df.Season.isin(seasons)].reset_index(drop=True)
                    if df.empty: st.error('Could not retrieve football data right now. Try again later; no result has been fabricated.')
                    elif not home.strip() or not away.strip() or home.strip().lower()==away.strip().lower(): st.warning('Enter two different team names.')
                    else:
                        teams=sorted(set(df.HomeTeam)|set(df.AwayTeam)); match={t.lower():t for t in teams}
                        h=match.get(home.strip().lower()); a=match.get(away.strip().lower())
                        if not h or not a:
                            st.warning('One or both team names were not found in the downloaded English Premier League historical dataset. Try a team spelling from this list: '+', '.join(teams))
                        else:
                            model,metrics=train_football(df); feat=pd.DataFrame([recent_features(df,h,a)]); proba=model.predict_proba(feat)[0]; classes=list(model.named_steps['model'].classes_); probs={c:float(p) for c,p in zip(classes,proba)}
                            labels={'H':h+' win','D':'Draw','A':a+' win'}; best=max(probs,key=probs.get)
                            st.markdown('<div class="panel"><div class="kicker">Prediction output · historical model</div><h2>'+labels.get(best,best)+'</h2><div class="muted">Estimated probabilities from the trained baseline. Not betting advice; not a guarantee.</div></div>',unsafe_allow_html=True)
                            cc=st.columns(3)
                            for c,k in zip(cc,['H','D','A']): c.metric(labels[k],f'{probs.get(k,0)*100:.1f}%')
                            st.markdown('#### Recent-form features used')
                            f=feat.iloc[0]; st.write(f"{h}: recent points-form proxy **{f.home_form:.2f}**, recent goal-difference average **{f.home_gd:.2f}**. {a}: form proxy **{f.away_form:.2f}**, goal-difference average **{f.away_gd:.2f}**.")
                            st.markdown('#### Temporal holdout evaluation (not this single prediction)')
                            if metrics:
                                mcols=st.columns(4)
                                for c,(k,v) in zip(mcols,metrics.items()): c.metric(k, f'{v:.3f}' if isinstance(v,float) else str(v))
                            st.caption(f'Data source: Football-Data.co.uk (English Premier League CSVs). Rows used: {len(df)}. The temporal split tests the latest 20% of rows; metrics are generated live when the app runs.')
                            st.markdown('[Open Football-Data.co.uk](https://www.football-data.co.uk/englandm.php)')
                except Exception as e: st.error(f'Prediction could not be completed: {e}. This is a runtime/data issue, not a prediction.')
    else:
        st.markdown('<div class="panel"><div class="kicker">02 / Match intelligence</div><h2>Cricket T20 outcome prediction</h2><div class="muted">Cricsheet historical T20 results → rolling team form → Logistic Regression baseline.</div></div>',unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1: t1=st.text_input('TEAM A','India',key='ct1')
        with c2: t2=st.text_input('TEAM B','Australia',key='ct2')
        if mode=='Advanced controls': st.info('Dataset currently covers available T20 international/domestic records in the Cricsheet T20 archive; the archive composition may change. Venue and toss are not used in this baseline.')
        if st.button('⚡ RUN CRICKET PREDICTION',use_container_width=True):
            with st.spinner('Downloading Cricsheet archive and training baseline… first run can take a little while.'):
                try:
                    df=load_cricket()
                    if df.empty: st.error('No usable T20 results were retrieved. Try later; no result has been fabricated.')
                    else:
                        allteams=sorted(set(df.team1)|set(df.team2)); lookup={x.lower():x for x in allteams}; a=lookup.get(t1.strip().lower()); b=lookup.get(t2.strip().lower())
                        if not a or not b: st.warning('Team not found in downloaded Cricsheet records. Available teams include: '+', '.join(allteams[:80]))
                        elif a==b: st.warning('Enter two different teams.')
                        else:
                            model,metrics=train_cricket(df); feat=pd.DataFrame([recent_features(df,a,b,cricket=True)]); p=model.predict_proba(feat)[0]; cls=list(model.named_steps['model'].classes_); probs={int(k):float(v) for k,v in zip(cls,p)}
                            st.markdown('<div class="panel"><div class="kicker">Prediction output · historical model</div><h2>'+ (a if probs.get(1,0)>=probs.get(0,0) else b)+' more likely to win</h2><div class="muted">Estimated probabilities from historical T20 outcomes. Not a guarantee.</div></div>',unsafe_allow_html=True)
                            c=st.columns(2); c[0].metric(a,f'{probs.get(1,0)*100:.1f}%'); c[1].metric(b,f'{probs.get(0,0)*100:.1f}%')
                            st.write(f'Latest rolling-form features: {a} **{feat.iloc[0].form_team1:.2f}**; {b} **{feat.iloc[0].form_team2:.2f}**. Matches parsed: **{len(df)}**.')
                            st.markdown('#### Temporal holdout evaluation')
                            if metrics:
                                mc=st.columns(4)
                                for c,(k,v) in zip(mc,metrics.items()): c.metric(k,f'{v:.3f}' if isinstance(v,float) else str(v))
                            st.caption('Source: Cricsheet downloadable T20 JSON archive. Metrics are computed on the latest 20% chronological holdout, then the deployed predictor is refit on available data.')
                            st.markdown('[Open Cricsheet](https://cricsheet.org/downloads/)')
                except Exception as e: st.error(f'Cricket prediction could not be completed: {e}. Check archive availability and retry.')
else:
    st.markdown('<div class="panel"><div class="kicker">03 / Fiction intelligence</div><h2>Character matchup lab</h2><div class="muted">Automatic character lookup using Wikipedia search and sourced summaries. The system does not pretend fictional win probabilities are scientifically calibrated.</div></div>',unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1: char1=st.text_input('CHARACTER A','Naruto Uzumaki')
    with c2: char2=st.text_input('CHARACTER B','Ichigo Kurosaki')
    if mode=='Advanced controls':
        version=st.selectbox('Character version',['Peak / strongest shown','Anime-only feats','Manga / canon feats','Early-series versions'])
        equal=st.checkbox('Equalize physical stats',value=False); prep=st.selectbox('Preparation',['No prep','1 hour prep','1 day prep']); battleground=st.selectbox('Battle conditions',['Neutral arena','Home universe','Random encounter'])
    else: version='Peak / strongest shown'; equal=False; prep='No prep'; battleground='Neutral arena'
    if st.button('⚡ ANALYSE FICTION MATCHUP',use_container_width=True):
        with st.spinner('Searching character references…'):
            try:
                results=[]
                for name in [char1,char2]:
                    hits=wiki_search(name+' fictional character')
                    chosen=next((x for x in hits if name.lower() in x.get('title','').lower()),hits[0] if hits else None)
                    summary=wiki_summary(chosen['title']) if chosen else None
                    results.append(summary)
                if not all(results): st.warning('Could not find reliable summaries for both names. Try a fuller character name or series title.')
                else:
                    for name,res in zip([char1,char2],results):
                        st.markdown(f'<div class="panel"><div class="kicker">Retrieved reference</div><h3>{res["title"]}</h3><div class="muted">{res["extract"][:1100] or "No introductory summary available."}</div><a href="{res["url"]}" target="_blank">Open source ↗</a></div>',unsafe_allow_html=True)
                    # No invented power stats. Show an explicitly non-calibrated neutral prior unless the user supplies a structured rubric.
                    st.markdown('<div class="panel"><div class="kicker">Scenario estimate</div><h2>Insufficient comparable evidence for a defensible automatic winner</h2><div class="muted">The retrieved biographies establish identity/context, not comparable combat feats. A neutral 50/50 prior is not a prediction and is deliberately not presented as a win probability.</div></div>',unsafe_allow_html=True)
                    st.write(f'Conditions: **{version}** · Equal physical stats: **{"Yes" if equal else "No"}** · Prep: **{prep}** · Arena: **{battleground}**.')
                    st.info('To make this module more capable without inventing facts, the next upgrade is a source-grounded feat database (feat, chapter/episode, source URL, evidence strength) plus an optional language-model API. Wikipedia alone cannot support objective power rankings across universes.')
            except Exception as e: st.error(f'Character lookup failed: {e}. Try again later.')

st.divider()
tab1,tab2,tab3=st.tabs(['RESEARCH EVIDENCE','DATA & SOURCES','SYSTEM LIMITS'])
with tab1:
    st.markdown('''### Reproducible evaluation design
- **Models:** Logistic Regression baseline with one-hot team identities and rolling recent-form features.
- **Split:** chronological 80/20 holdout; latest matches are held out from model fitting for evaluation.
- **Metrics:** accuracy, macro-F1 and log loss for football; accuracy, macro-F1 and Brier score for cricket.
- **Leakage control:** each match's rolling-form features are computed using earlier rows only.
- **Reproducibility:** dataset source and model family are fixed in this code; downloaded data may update, so archive snapshots and retrieval dates should be saved for final thesis experiments.

**Important:** metrics shown after a successful run are generated from the retrieved dataset at runtime. No performance number is hard-coded into this app.''')
with tab2:
    st.markdown('''- Football historical results: [Football-Data.co.uk](https://www.football-data.co.uk/englandm.php)
- Cricket historical results: [Cricsheet downloads](https://cricsheet.org/downloads/)
- Fictional character summaries: [Wikipedia API](https://en.wikipedia.org/w/api.php)
- Machine learning: scikit-learn Logistic Regression.

Sources are linked so the user can inspect them. The app currently does not fetch live odds, real-time injury reports, or live match scores.''')
with tab3:
    st.markdown('''- Football baseline currently uses English Premier League results, not every global league.
- Cricket baseline uses historical Cricsheet T20 records; data coverage depends on the archive.
- Predictions are historical-model estimates and may be unreliable for teams with little/no history.
- Fictional summaries are retrieved automatically, but the app intentionally does not fabricate feats or present subjective comparisons as calibrated probabilities.
- This is a functioning research prototype, not a completed or independently validated thesis result. Save dated data snapshots, test results and source details before making academic claims.''')

st.markdown('<div style="text-align:center;color:#637894;font-size:.75rem;padding:18px 0">RAN-NEETI · Research prototype · Transparent by design · No fabricated metrics</div>',unsafe_allow_html=True)
