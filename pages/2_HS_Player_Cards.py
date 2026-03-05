import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="HS Player Cards", page_icon="🏫", layout="wide")
st.title("🏫 High School Player Cards")

BG_COLOR='#0d0d0d'; PANEL_COLOR='#161616'; BORDER_COLOR='#2a2a2a'
TEXT_COLOR='#e8e8e8'; DIM_COLOR='#888888'

STAT_DEFS = [
    ('avg','AVG',True,'.3f'),('obp','OBP',True,'.3f'),('slg','SLG',True,'.3f'),
    ('ops','OPS',True,'.3f'),('woba','wOBA',True,'.3f'),('iso','ISO',True,'.3f'),
    ('hr','HR',True,'d'),('rbi','RBI',True,'d'),
    ('r','R',True,'d'),('h','H',True,'d'),('2b','2B',True,'d'),('3b','3B',True,'d'),
    ('sb','SB',True,'d'),('bb','BB',True,'d'),('k','K',False,'d'),
    ('kpct','K%',False,'.1f'),('bbpct','BB%',True,'.1f'),
]

DATA_PATH = 'data/hitters.csv'
LEADERBOARD_STATS = ['avg','obp','slg','ops','woba','iso','hr','rbi','sb']
LEADERBOARD_LABELS = {'avg':'AVG','obp':'OBP','slg':'SLG','ops':'OPS',
                      'woba':'wOBA','iso':'ISO','hr':'HR','rbi':'RBI','sb':'SB'}

def pct_color(pct):
    if pct >= 50:
        t=(pct-50)/50; r=int(255*t+220*(1-t)); g=int(50*t+220*(1-t)); b=int(50*t+220*(1-t))
    else:
        t=pct/50; r=int(220*t+40*(1-t)); g=int(220*t+80*(1-t)); b=int(220*t+255*(1-t))
    return f"#{r:02x}{g:02x}{b:02x}"

def get_percentile(value, series, higher_is_better=True):
    s=series.dropna()
    if len(s)==0: return 50
    return float(np.sum(s<=value)/len(s)*100) if higher_is_better else float(np.sum(s>=value)/len(s)*100)

def compute_stats(df):
    d=df.copy(); d.columns=[c.strip().lower() for c in d.columns]
    for col in ['hbp','sf','2b','3b','hr','bb','k','sb','r','rbi']:
        if col not in d.columns: d[col]=0
    d=d.fillna(0)
    for col in ['ab','h','2b','3b','hr','r','rbi','bb','k','sb','hbp','sf']:
        d[col]=pd.to_numeric(d[col],errors='coerce').fillna(0).astype(int)
    d['1b']=d['h']-d['2b']-d['3b']-d['hr']
    d['avg']=np.where(d['ab']>0,d['h']/d['ab'],0)
    obp_num=d['h']+d['bb']+d['hbp']; obp_den=d['ab']+d['bb']+d['hbp']+d['sf']
    d['obp']=np.where(obp_den>0,obp_num/obp_den,0)
    slg_num=d['1b']+2*d['2b']+3*d['3b']+4*d['hr']
    d['slg']=np.where(d['ab']>0,slg_num/d['ab'],0)
    d['ops']=d['obp']+d['slg']
    d['iso']=d['slg']-d['avg']
    pa=d['ab']+d['bb']+d['hbp']+d['sf']
    woba_num=(0.69*d['bb'])+(0.72*d['hbp'])+(0.88*d['1b'])+(1.25*d['2b'])+(1.59*d['3b'])+(2.05*d['hr'])
    d['woba']=np.where(pa>0,woba_num/pa,0)
    d['kpct']=np.where(pa>0,d['k']/pa*100,0)
    d['bbpct']=np.where(pa>0,d['bb']/pa*100,0)
    d['pa']=pa
    d['player_norm']=d['player'].str.strip().str.lower()
    return d

@st.cache_data
def load_data():
    try:
        raw=pd.read_csv(DATA_PATH)
        return compute_stats(raw)
    except FileNotFoundError:
        return None

def build_card(p, qual, min_ab, selected_year, all_years):
    fig=plt.figure(figsize=(20,14),facecolor=BG_COLOR)
    outer=gridspec.GridSpec(3,1,figure=fig,height_ratios=[2.8,5.5,5.5],
                            hspace=0.18,left=0.04,right=0.96,top=0.97,bottom=0.03)
    ax_hdr=fig.add_subplot(outer[0]); ax_hdr.set_facecolor(PANEL_COLOR); ax_hdr.axis('off')
    ax_hdr.set_xlim(0,1); ax_hdr.set_ylim(0,1)
    ax_hdr.plot([0,1],[0.997,0.997],color='#cc0000',linewidth=4,transform=ax_hdr.transAxes,clip_on=False)
    ax_hdr.text(0.01,0.72,p['player'].upper(),transform=ax_hdr.transAxes,color=TEXT_COLOR,fontsize=22,fontweight='bold',va='center')
    ax_hdr.text(0.01,0.25,f"{p['team']}  ·  {p['pos']}  ·  {selected_year}",
                transform=ax_hdr.transAxes,color=DIM_COLOR,fontsize=11,va='center')
    is_qual=p['ab']>=min_ab
    pool_years=f"{min(all_years)}–{max(all_years)}" if len(all_years)>1 else str(all_years[0])
    ax_hdr.text(0.01,0.0,
                f"{'QUALIFIED' if is_qual else 'NOT QUALIFIED'}  ({int(p['ab'])} AB)  ·  Percentiles vs {len(qual)}-player pool ({pool_years})",
                transform=ax_hdr.transAxes,color='#2ecc71' if is_qual else '#e74c3c',fontsize=9,fontweight='bold',va='bottom')
    keys=['avg','obp','slg','ops','woba','iso','hr','rbi','sb','ab']
    labels={'avg':'AVG','obp':'OBP','slg':'SLG','ops':'OPS','woba':'wOBA','iso':'ISO','hr':'HR','rbi':'RBI','sb':'SB','ab':'AB'}
    n_boxes=len(keys); bx0=0.30; bw=0.68; bh=0.88; by0=0.06; cw=bw/n_boxes
    ax_hdr.add_patch(FancyBboxPatch((bx0,by0),bw,bh,boxstyle='round,pad=0.005',
        facecolor='#1a1a1a',edgecolor=BORDER_COLOR,linewidth=1.2,transform=ax_hdr.transAxes,zorder=2))
    for i,key in enumerate(keys):
        cx=bx0+cw*(i+0.5); val=p.get(key,0)
        val_str=(f"{float(val):.3f}".lstrip('0') if key in ['avg','obp','slg','ops','woba','iso'] and float(val)<1
                 else f"{float(val):.3f}" if key in ['avg','obp','slg','ops','woba','iso']
                 else str(int(float(val))))
        sd=next((s for s in STAT_DEFS if s[0]==key),None)
        col=(pct_color(get_percentile(float(val),qual[key],sd[2])) if sd and key in qual.columns and len(qual)>0 else TEXT_COLOR)
        ax_hdr.text(cx,by0+bh*0.68,val_str,transform=ax_hdr.transAxes,color=col,fontsize=15,fontweight='bold',ha='center',va='center',zorder=3)
        ax_hdr.text(cx,by0+bh*0.22,labels[key],transform=ax_hdr.transAxes,color='#cccccc',fontsize=10,ha='center',va='center',zorder=3)
        if i<n_boxes-1:
            ax_hdr.plot([bx0+cw*(i+1)]*2,[by0+0.06,by0+bh-0.06],color=BORDER_COLOR,linewidth=0.7,transform=ax_hdr.transAxes,zorder=3)

    def draw_pct_section(ax, stat_keys):
        ax.set_facecolor(PANEL_COLOR); ax.axis('off'); ax.set_xlim(0,1); ax.set_ylim(0,1)
        n=len(stat_keys); row_h=0.92/n; y_start=0.97
        for i,key in enumerate(stat_keys):
            sd=next((s for s in STAT_DEFS if s[0]==key),None)
            if not sd: continue
            _,label,higher,fmt=sd; val=float(p.get(key,0))
            pct=get_percentile(val,qual[key],higher) if key in qual.columns and len(qual)>0 else 50
            color=pct_color(pct); y=y_start-row_h*(i+0.5)
            bl=0.18; br=0.78; bw2=br-bl; bh2=row_h*0.32; by2=y-bh2/2
            ax.text(bl-0.02,y,label,ha='right',va='center',color=DIM_COLOR,fontsize=11,fontweight='bold',transform=ax.transAxes)
            ax.add_patch(FancyBboxPatch((bl,by2),bw2,bh2,boxstyle='round,pad=0.002',facecolor='#252525',edgecolor='none',transform=ax.transAxes,zorder=1))
            ax.add_patch(FancyBboxPatch((bl,by2),max(bw2*pct/100,0.004),bh2,boxstyle='round,pad=0.002',facecolor=color,edgecolor='none',transform=ax.transAxes,zorder=2))
            ax.plot([bl+bw2*0.5]*2,[by2-0.005,by2+bh2+0.005],color='#555555',linewidth=1.0,transform=ax.transAxes,zorder=3)
            val_str=(f"{val:.3f}".lstrip('0') if fmt=='.3f' and val<1
                     else f"{val:.3f}" if fmt=='.3f'
                     else f"{val:.1f}%" if fmt=='.1f'
                     else str(int(val)))
            ax.text(br+0.02,y,val_str,ha='left',va='center',color=color,fontsize=11,fontweight='bold',transform=ax.transAxes)
            ax.text(0.99,y,f"{int(pct)}th",ha='right',va='center',color=DIM_COLOR,fontsize=9,transform=ax.transAxes)
        ax.text(0.01,0.99,f'PERCENTILE RANKINGS  (vs {pool_years} league pool)',
                ha='left',va='top',color=TEXT_COLOR,fontsize=10,fontweight='bold',transform=ax.transAxes)

    body_gs=gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[1],wspace=0.12)
    draw_pct_section(fig.add_subplot(body_gs[0]),['avg','obp','slg','ops','woba','iso','hr','rbi','r','h'])
    draw_pct_section(fig.add_subplot(body_gs[1]),['2b','3b','sb','bb','k','kpct','bbpct'])

    ax_tbl=fig.add_subplot(outer[2]); ax_tbl.set_facecolor(PANEL_COLOR); ax_tbl.axis('off')
    ax_tbl.set_xlim(0,1); ax_tbl.set_ylim(0,1)
    tbl_cols=['ab','h','1b','2b','3b','hr','r','rbi','bb','k','sb','hbp','pa']
    tbl_labels=['AB','H','1B','2B','3B','HR','R','RBI','BB','K','SB','HBP','PA']
    cw3=1.0/len(tbl_cols)
    for ci,lbl in enumerate(tbl_labels):
        ax_tbl.text((ci+0.5)*cw3,0.88,lbl,ha='center',va='center',color='#cccccc',fontsize=12,fontweight='bold',transform=ax_tbl.transAxes)
    ax_tbl.plot([0.01,0.99],[0.72,0.72],color=BORDER_COLOR,linewidth=0.8,transform=ax_tbl.transAxes)
    for ci,key in enumerate(tbl_cols):
        val=p.get(key,0); sd=next((s for s in STAT_DEFS if s[0]==key),None)
        col=(pct_color(get_percentile(float(val),qual[key],sd[2])) if sd and key in qual.columns and len(qual)>0 else TEXT_COLOR)
        ax_tbl.text((ci+0.5)*cw3,0.42,str(int(float(val))),ha='center',va='center',color=col,fontsize=14,fontweight='bold',transform=ax_tbl.transAxes)
    ax_tbl.plot([0.01,0.99],[0.28,0.28],color=BORDER_COLOR,linewidth=0.5,transform=ax_tbl.transAxes)
    ax_tbl.text(0.01,0.18,'LG AVG',ha='left',va='center',color=DIM_COLOR,fontsize=9,transform=ax_tbl.transAxes)
    for ci,key in enumerate(tbl_cols):
        if key in qual.columns:
            ax_tbl.text((ci+0.5)*cw3,0.12,f"{qual[key].mean():.1f}",ha='center',va='center',color=DIM_COLOR,fontsize=9,transform=ax_tbl.transAxes)
    ax_tbl.text(0.01,0.97,f'{selected_year} SEASON STATS',ha='left',va='top',color=TEXT_COLOR,fontsize=11,fontweight='bold',transform=ax_tbl.transAxes)
    ax_tbl.text(0.99,0.97,f"League pool: {len(qual)} qualified players · {pool_years} · min {min_ab} AB",
                ha='right',va='top',color=DIM_COLOR,fontsize=9,transform=ax_tbl.transAxes)
    fig.text(0.5,0.005,'Stats via MaxPreps  ·  Percentiles vs multi-year qualified league pool',
             ha='center',va='bottom',color='#555555',fontsize=9,style='italic')

    buf=BytesIO()
    plt.savefig(buf,dpi=180,bbox_inches='tight',facecolor=BG_COLOR,edgecolor='none',format='png')
    plt.close(fig); buf.seek(0)
    return buf

# ── Load data ─────────────────────────────────────────────────────────────────
df=load_data()
if df is None:
    st.error("No data file found. Make sure `data/hitters.csv` is in the repo.")
    st.stop()

all_years=sorted(df['year'].astype(int).unique().tolist())
min_ab=int(df['ab'].quantile(0.40))
qual=df[df['ab']>=min_ab].copy()
pool_years=f"{min(all_years)}–{max(all_years)}" if len(all_years)>1 else str(all_years[0])

# ── LEADERBOARD ───────────────────────────────────────────────────────────────
st.markdown("### 🏆 Leaderboard")
lb_col1,lb_col2=st.columns(2)
with lb_col1:
    lb_stat=st.selectbox("Sort by",LEADERBOARD_STATS,format_func=lambda x: LEADERBOARD_LABELS[x])
with lb_col2:
    lb_year=st.selectbox("Season",["All years"]+[str(y) for y in sorted(all_years,reverse=True)])

lb_df=qual.copy()
if lb_year != "All years":
    lb_df=lb_df[lb_df['year'].astype(int)==int(lb_year)]

if not lb_df.empty and lb_stat in lb_df.columns:
    top=lb_df.nlargest(10,lb_stat)[['player','team','year',lb_stat,'ab']].copy()
    top['year']=top['year'].astype(int)
    fmt=next((s[3] for s in STAT_DEFS if s[0]==lb_stat),'.3f')
    if fmt=='.3f':
        top[lb_stat]=top[lb_stat].apply(lambda x: f"{x:.3f}".lstrip('0') if x<1 else f"{x:.3f}")
    elif fmt=='.1f':
        top[lb_stat]=top[lb_stat].apply(lambda x: f"{x:.1f}%")
    else:
        top[lb_stat]=top[lb_stat].apply(lambda x: str(int(x)))
    top.columns=['Player','Team','Year',LEADERBOARD_LABELS[lb_stat],'AB']
    top.index=range(1,len(top)+1)
    st.dataframe(top,use_container_width=True)

st.markdown("---")

# ── PLAYER SEARCH ─────────────────────────────────────────────────────────────
st.markdown("### 🔍 Player Card")
col1,col2=st.columns([2,1])
with col1:
    search=st.selectbox("Search player",[""] + sorted(df['player'].unique().tolist()))
with col2:
    if search:
        player_years=sorted(df[df['player_norm']==search.strip().lower()]['year'].astype(int).unique().tolist(),reverse=True)
        selected_year=st.selectbox("Season",["Select a year..."]+[str(y) for y in player_years])
    else:
        st.selectbox("Season",["—"],disabled=True)
        selected_year="Select a year..."

if search and selected_year not in ["Select a year...","—",""]:
    season_row=df[(df['player_norm']==search.strip().lower())&(df['year'].astype(int)==int(selected_year))]
    if season_row.empty:
        st.warning("No data found for that player/season.")
    else:
        p=season_row.iloc[0]
        with st.spinner("Building card..."):
            buf=build_card(p,qual,min_ab,selected_year,all_years)
        st.image(buf,use_column_width=True)
        st.download_button("⬇ Download Card PNG",data=buf,
                           file_name=f"{search.replace(' ','_')}_{selected_year}_card.png",
                           mime="image/png")
