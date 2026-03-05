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
    ('ops','OPS',True,'.3f'),('hr','HR',True,'d'),('rbi','RBI',True,'d'),
    ('r','R',True,'d'),('h','H',True,'d'),('2b','2B',True,'d'),('3b','3B',True,'d'),
    ('sb','SB',True,'d'),('bb','BB',True,'d'),('k','K',False,'d'),
    ('kpct','K%',False,'.1f'),('bbpct','BB%',True,'.1f'),
]

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
    pa=d['ab']+d['bb']+d['hbp']+d['sf']
    d['kpct']=np.where(pa>0,d['k']/pa*100,0)
    d['bbpct']=np.where(pa>0,d['bb']/pa*100,0)
    d['pa']=pa
    d['player_norm']=d['player'].str.strip().str.lower()
    return d

def aggregate_player(df, name):
    p=df[df['player_norm']==name.strip().lower()].copy()
    if p.empty: return None
    agg={'player':p['player'].iloc[0],'team':p['team'].iloc[-1],
         'pos':p['pos'].iloc[-1],'years':sorted(p['year'].astype(int).tolist())}
    agg['year_str']=' / '.join(str(y) for y in agg['years'])
    for col in ['ab','h','2b','3b','hr','r','rbi','bb','k','sb','hbp','sf']:
        agg[col]=int(p[col].sum())
    tmp=pd.DataFrame([agg]); tmp=compute_stats(tmp); row=tmp.iloc[0]
    for stat in ['avg','obp','slg','ops','kpct','bbpct','pa','1b']:
        agg[stat]=row[stat]
    return pd.Series(agg)

def build_card(p, qual, min_ab):
    fig=plt.figure(figsize=(20,14),facecolor=BG_COLOR)
    outer=gridspec.GridSpec(3,1,figure=fig,height_ratios=[2.8,5.5,5.5],
                            hspace=0.18,left=0.04,right=0.96,top=0.97,bottom=0.03)

    # HEADER
    ax_hdr=fig.add_subplot(outer[0]); ax_hdr.set_facecolor(PANEL_COLOR); ax_hdr.axis('off')
    ax_hdr.set_xlim(0,1); ax_hdr.set_ylim(0,1)
    ax_hdr.plot([0,1],[0.997,0.997],color='#cc0000',linewidth=4,transform=ax_hdr.transAxes,clip_on=False)
    ax_hdr.text(0.01,0.72,p['player'].upper(),transform=ax_hdr.transAxes,color=TEXT_COLOR,fontsize=22,fontweight='bold',va='center')
    ax_hdr.text(0.01,0.25,f"{p['team']}  ·  {p['pos']}  ·  {p['year_str']}",transform=ax_hdr.transAxes,color=DIM_COLOR,fontsize=11,va='center')
    is_qual=p['ab']>=min_ab
    ax_hdr.text(0.01,0.0,f"{'QUALIFIED' if is_qual else 'NOT QUALIFIED'}  ({int(p['ab'])} AB{'' if is_qual else f' / {min_ab} req'})",
                transform=ax_hdr.transAxes,color='#2ecc71' if is_qual else '#e74c3c',fontsize=9,fontweight='bold',va='bottom')

    keys=['avg','obp','slg','ops','hr','rbi','sb','ab']
    labels={'avg':'AVG','obp':'OBP','slg':'SLG','ops':'OPS','hr':'HR','rbi':'RBI','sb':'SB','ab':'AB'}
    n_boxes=len(keys); bx0=0.30; bw=0.68; bh=0.88; by0=0.06; cw=bw/n_boxes
    ax_hdr.add_patch(FancyBboxPatch((bx0,by0),bw,bh,boxstyle='round,pad=0.005',
        facecolor='#1a1a1a',edgecolor=BORDER_COLOR,linewidth=1.2,transform=ax_hdr.transAxes,zorder=2))
    for i,key in enumerate(keys):
        cx=bx0+cw*(i+0.5); val=p.get(key,0)
        val_str=f"{float(val):.3f}".lstrip('0') if key in ['avg','obp','slg','ops'] and float(val)<1 else (f"{float(val):.3f}" if key in ['avg','obp','slg','ops'] else str(int(float(val))))
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
            val_str=(f"{val:.3f}".lstrip('0') if fmt=='.3f' and val<1 else f"{val:.1f}%" if fmt=='.1f' else str(int(val)))
            ax.text(br+0.02,y,val_str,ha='left',va='center',color=color,fontsize=11,fontweight='bold',transform=ax.transAxes)
            ax.text(0.99,y,f"{int(pct)}th",ha='right',va='center',color=DIM_COLOR,fontsize=9,transform=ax.transAxes)
        ax.text(0.01,0.99,'PERCENTILE RANKINGS  (vs qualified league hitters)',ha='left',va='top',color=TEXT_COLOR,fontsize=10,fontweight='bold',transform=ax.transAxes)

    body_gs=gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[1],wspace=0.12)
    draw_pct_section(fig.add_subplot(body_gs[0]),['avg','obp','slg','ops','hr','rbi','r','h'])
    draw_pct_section(fig.add_subplot(body_gs[1]),['2b','3b','sb','bb','k','kpct','bbpct'])

    # COUNTING STATS TABLE
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
    ax_tbl.text(0.01,0.97,'CAREER COUNTING STATS',ha='left',va='top',color=TEXT_COLOR,fontsize=11,fontweight='bold',transform=ax_tbl.transAxes)
    ax_tbl.text(0.99,0.97,f"League pool: {len(qual)} qualified players (min {min_ab} AB)",ha='right',va='top',color=DIM_COLOR,fontsize=9,transform=ax_tbl.transAxes)
    fig.text(0.5,0.005,'Stats via MaxPreps  ·  Percentiles vs qualified league hitters',ha='center',va='bottom',color='#555555',fontsize=9,style='italic')

    buf=BytesIO()
    plt.savefig(buf,dpi=180,bbox_inches='tight',facecolor=BG_COLOR,edgecolor='none',format='png')
    plt.close(fig); buf.seek(0)
    return buf

# ── UI ────────────────────────────────────────────────────────────────────────
uploaded=st.file_uploader("Upload league stats CSV",type="csv")

with st.expander("📋 CSV Format"):
    st.markdown("Columns: `player, team, year, pos, ab, h, 2b, 3b, hr, r, rbi, bb, k, sb, hbp, sf` — one row per player per season.")
    template=pd.DataFrame([{'player':'John Smith','team':'Lincoln High','year':2024,'pos':'SS','ab':87,'h':31,'2b':6,'3b':1,'hr':3,'r':22,'rbi':18,'bb':12,'k':14,'sb':8,'hbp':2,'sf':1}])
    st.download_button("⬇ Download template",data=template.to_csv(index=False),file_name="template.csv",mime="text/csv")

if uploaded:
    try:
        raw=pd.read_csv(uploaded); df=compute_stats(raw)
        all_players=df['player_norm'].unique()
        career_rows=[aggregate_player(df,n) for n in all_players]
        career_df=pd.DataFrame(career_rows).reset_index(drop=True)
        st.success(f"{len(career_df)} players loaded")
        min_ab=st.slider("Minimum AB to qualify",10,int(career_df['ab'].max()),int(career_df['ab'].quantile(0.40)),5)
        qual=career_df[career_df['ab']>=min_ab].copy()
        st.caption(f"{len(qual)} of {len(career_df)} players qualify at {min_ab} AB")
        search=st.selectbox("Search player",[""] + sorted(career_df['player'].tolist()))
        if search:
            row=career_df[career_df['player']==search].iloc[0]
            with st.spinner("Building card..."):
                buf=build_card(row,qual,min_ab)
            st.image(buf,use_column_width=True)
            st.download_button("⬇ Download Card PNG",data=buf,file_name=f"{search.replace(' ','_')}_card.png",mime="image/png")
    except Exception as e:
        st.error(f"Error: {e}"); st.exception(e)
else:
    st.info("Upload your league CSV above to get started.")
