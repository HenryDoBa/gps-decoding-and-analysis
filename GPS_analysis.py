#!/usr/bin/env python3
"""Reproducible NMEA 0183 GPS analysis for the two course recordings.

Usage:
  python gps_analysis.py nmea-test1-1.txt nmea-test2-1.txt --out-dir gps_results

Reads NMEA GGA, RMC, GSA, VTG and GSV, validates checksums, reports basic
statistics, and generates trajectory, speed, GPS-quality and scatter plots.
"""
import argparse
import csv
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EARTH_RADIUS_M = 6371000.0

def nmea_checksum_ok(sentence):
    if '*' not in sentence:
        return False
    payload, after_star = sentence[1:].split('*', 1)
    if not after_star or len(after_star) < 2:
        return False
    check = 0
    for ch in payload:
        check ^= ord(ch)
    try:
        return check == int(after_star[:2], 16)
    except ValueError:
        return False

def decode_latlon(raw, hemi):
    if not raw or not hemi:
        return math.nan
    n = float(raw)
    degrees = math.floor(n / 100)
    minutes = n - 100 * degrees
    value = degrees + minutes / 60
    return -value if hemi in ('S', 'W') else value

def parse_time(value):
    if not value:
        return math.nan
    try:
        h, m, s = int(value[:2]), int(value[2:4]), float(value[4:])
        return h*3600 + m*60 + s
    except Exception:
        return math.nan

def numeric(fields, idx):
    try:
        return float(fields[idx]) if fields[idx] else math.nan
    except (IndexError, ValueError):
        return math.nan

def parse_file(path):
    """Return one record per RMC epoch with same-time GGA/GSA from sequential NMEA burst."""
    records, quality, counter = [], {}, Counter()
    total, valid_checksum, invalid_checksum, incomplete = 0, 0, 0, 0
    date_texts = []
    latest_gsa = {}
    latest_gga = {}
    for raw_line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        line = raw_line.strip().rstrip('\\').strip()
        if not line.startswith('$GP') or '*' not in line:
            continue
        total += 1
        if not nmea_checksum_ok(line):
            invalid_checksum += 1
            continue
        valid_checksum += 1
        fields = line[1:line.index('*')].split(',')
        msg = fields[0]
        counter[msg] += 1
        if msg == 'GPGGA':
            latest_gga = dict(t=parse_time(fields[1]), lat=decode_latlon(fields[2], fields[3]),
                              lon=decode_latlon(fields[4], fields[5]), fix=int(fields[6]) if fields[6] else 0,
                              n_sat=int(fields[7]) if fields[7] else 0,
                              hdop=numeric(fields,8), altitude=numeric(fields,9), geoid=numeric(fields,11))
        elif msg == 'GPGSA':
            latest_gsa = dict(fix_dimension=int(fields[2]) if fields[2] else 0,
                              prns=[v for v in fields[3:15] if v],
                              pdop=numeric(fields,15), hdop_gsa=numeric(fields,16), vdop=numeric(fields,17))
        elif msg == 'GPGSV':
            if len(fields) >= 4:
                try:
                    quality['satellites_in_view_max'] = max(int(fields[3]), quality.get('satellites_in_view_max', 0))
                    quality['satellites_in_view_min'] = min(int(fields[3]), quality.get('satellites_in_view_min', int(fields[3])))
                except ValueError:
                    pass
        elif msg == 'GPRMC':
            rmc = dict(t=parse_time(fields[1]), status=fields[2],
                       lat=decode_latlon(fields[3],fields[4]), lon=decode_latlon(fields[5],fields[6]),
                       speed_knots=numeric(fields,7), course_deg=numeric(fields,8), date=fields[9],
                       mode=fields[12] if len(fields)>12 else '')
            if rmc['date']:
                date_texts.append(rmc['date'])
            if latest_gga and abs(latest_gga['t'] - rmc['t']) < 0.001:
                rmc.update({'gga_'+k:v for k,v in latest_gga.items() if k not in ['lat','lon','t']})
            else:
                incomplete += 1
            if latest_gsa:
                rmc.update({'gsa_'+k:v for k,v in latest_gsa.items()})
            records.append(rmc)
    return records, counter, dict(quality, nmea_lines=total,
                                  valid_checksum=valid_checksum, invalid_checksum=invalid_checksum,
                                  unmatched_gga=incomplete, dates=sorted(set(date_texts)))

def haversine(lat1, lon1, lat2, lon2):
    a1, a2 = math.radians(lat1), math.radians(lat2)
    dlat, dlon = a2-a1, math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(a1)*math.cos(a2)*math.sin(dlon/2)**2
    return 2*EARTH_RADIUS_M*math.asin(min(1, math.sqrt(a)))

def local_xy(lat, lon, latitude0, longitude0):
    lat, lon = np.asarray(lat,dtype=float), np.asarray(lon,dtype=float)
    east = EARTH_RADIUS_M * np.radians(lon-longitude0) * math.cos(math.radians(latitude0))
    north = EARTH_RADIUS_M * np.radians(lat-latitude0)
    return east, north

def fmt(v, digits=2):
    return f'{v:.{digits}f}' if math.isfinite(v) else 'N/A'

def metrics(records, subsets=None):
    valid = [r for r in records if r['status']=='A' and np.isfinite(r['lat']) and np.isfinite(r['lon']) and r.get('gga_fix',0)>0]
    if subsets is not None:
        valid = [r for r in valid if subsets(r)]
    if not valid:
        return {}, valid
    lats = np.array([r['lat'] for r in valid]); lons=np.array([r['lon'] for r in valid]);
    ts=np.array([r['t'] for r in valid]);
    speeds=np.array([r['speed_knots'] for r in valid]);
    hds=np.array([r.get('gga_hdop',math.nan) for r in valid])
    alt=np.array([r.get('gga_altitude',math.nan) for r in valid]);
    sats=np.array([r.get('gga_n_sat',math.nan) for r in valid]);
    lat0,lon0=np.mean(lats),np.mean(lons)
    east,north=local_xy(lats,lons,lat0,lon0)
    radial=np.hypot(east,north)
    segment_dist = np.array([haversine(*p1,*p2) for p1,p2 in zip(zip(lats[:-1],lons[:-1]),zip(lats[1:],lons[1:]))])
    net_dist=haversine(lats[0],lons[0],lats[-1],lons[-1]); path_len=segment_dist.sum()
    stats=dict(n=len(valid), duration_s=ts[-1]-ts[0], start_s=ts[0], end_s=ts[-1],
               start_lat=lats[0],start_lon=lons[0],end_lat=lats[-1],end_lon=lons[-1],
               lat_mean=lat0,lon_mean=lon0,
               min_speed_kmh=np.nanmin(speeds)*1.852, mean_speed_kmh=np.nanmean(speeds)*1.852,
               median_speed_kmh=np.nanmedian(speeds)*1.852,max_speed_kmh=np.nanmax(speeds)*1.852,
               p95_speed_kmh=np.nanpercentile(speeds,95)*1.852,
               speed_under_1kmh_pct=100*np.mean(speeds*1.852<1),
               sat_min=np.nanmin(sats),sat_max=np.nanmax(sats),sat_mean=np.nanmean(sats),
               hdop_median=np.nanmedian(hds),hdop_max=np.nanmax(hds),
               altitude_median=np.nanmedian(alt),altitude_min=np.nanmin(alt),altitude_max=np.nanmax(alt),
               net_displacement_m=net_dist, successive_distance_m=path_len,
               e_std_m=np.std(east,ddof=1) if len(east)>1 else math.nan,
               n_std_m=np.std(north,ddof=1) if len(east)>1 else math.nan,
               radial_rms_m=np.sqrt(np.mean(radial**2)),
               radial_p50_m=np.percentile(radial,50),radial_p68_m=np.percentile(radial,68),
               radial_p95_m=np.percentile(radial,95),radial_max_m=np.max(radial),
               radial_2drms_m=2*np.sqrt(np.std(east,ddof=1)**2+np.std(north,ddof=1)**2) if len(east)>1 else math.nan,
               e_span_m=np.ptp(east),n_span_m=np.ptp(north))
    return stats, valid

def clock(seconds):
    i = int(seconds)
    return f'{i//3600:02d}:{i//60%60:02d}:{i%60:02d}'

def format_stats(stats):
    return '\n'.join([f'  {k:25s}: {fmt(v) if isinstance(v,(float, np.floating)) else v}' for k,v in stats.items()])

def plot_trajectories(data, outdir):
    valid1, valid2=data
    both = valid1+valid2
    lat0=np.mean([r['lat'] for r in both]);lon0=np.mean([r['lon'] for r in both]);
    fig, axs=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for i,valid in enumerate([valid1,valid2]):
        e,n=local_xy([r['lat'] for r in valid],[r['lon'] for r in valid],lat0,lon0)
        t=np.array([r['t'] for r in valid]);
        sc=axs[i].scatter(e,n,c=t-t[0],s=10,cmap='viridis')
        axs[i].plot(e,n,lw=0.75,color='0.5',alpha=0.6)
        axs[i].scatter(e[0],n[0],s=70,marker='o',facecolor='white',edgecolor='black',label='Start',zorder=4)
        axs[i].scatter(e[-1],n[-1],s=70,marker='X',color='red',label='End',zorder=4)
        axs[i].set(title=f'Test {i+1}: GPS trajectory ({len(valid)} valid fixes)',xlabel='East [m]',ylabel='North [m]')
        axs[i].grid(alpha=0.25);axs[i].axis('equal');axs[i].legend(fontsize=8)
        fig.colorbar(sc,ax=axs[i],label='Elapsed time [s]',shrink=0.85)
    fig.savefig(outdir/'trajectories.png',dpi=200)
    plt.close(fig)

def plot_speeds(data,outdir):
    fig,ax=plt.subplots(figsize=(11,3.8),constrained_layout=True)
    for i,valid in enumerate(data):
        ts=np.array([r['t'] for r in valid]);speeds=np.array([r['speed_knots'] for r in valid])*1.852
        ax.plot(ts-ts[0],speeds,label=f'Test {i+1}',lw=1.5)
    ax.set(xlabel='Elapsed time since first valid fix [s]',ylabel='Speed from RMC [km/h]',title='Receiver-reported speed comparison')
    ax.grid(alpha=0.3);ax.legend();fig.savefig(outdir/'speeds.png',dpi=200);plt.close(fig)

def plot_stationary(valid, outdir, selected_label):
    lat0=np.mean([r['lat'] for r in valid]);lon0=np.mean([r['lon'] for r in valid])
    e,n=local_xy([r['lat'] for r in valid],[r['lon'] for r in valid],lat0,lon0)
    r=np.hypot(e,n);r95=np.percentile(r,95)
    fig,ax=plt.subplots(figsize=(6,6),constrained_layout=True)
    c=ax.scatter(e,n,s=15,c=np.array([r['t'] for r in valid])-valid[0]['t'],cmap='viridis')
    ax.scatter(0,0,marker='+',s=150,color='red',label='Mean GPS position')
    ax.add_patch(plt.Circle((0,0),r95,fill=False,edgecolor='red',linestyle='--',lw=1.5,label=f'95% empirical radius: {r95:.2f} m'))
    ax.set(xlabel='East relative to mean [m]',ylabel='North relative to mean [m]',title=selected_label)
    ax.grid(alpha=0.25);ax.legend(loc='upper right',fontsize=8)
    ax.axis('equal');fig.colorbar(c,ax=ax,label='Elapsed time [s]')
    fig.savefig(outdir/'stationary_scatter.png',dpi=200);plt.close(fig)

def export_csv(records,path):
    fields=['t','date','lat','lon','status','speed_knots','course_deg','mode','gga_fix','gga_n_sat','gga_hdop','gga_altitude','gga_geoid','gsa_fix_dimension','gsa_pdop','gsa_hdop_gsa','gsa_vdop']
    with open(path,'w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(records)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('file1',type=Path)
    ap.add_argument('file2',type=Path)
    ap.add_argument('--out-dir',default='gps_results',type=Path)
    args=ap.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=True)
    all_valid=[]
    for i,path in enumerate([args.file1,args.file2],1):
        records,counts,quality=parse_file(path)
        stats,valid=metrics(records)
        all_valid.append(valid)
        print(f'\nFILE {i} ({path.name})')
        print('  Sentence types:',dict(counts))
        print('  Checksum/log info:',quality)
        print('  RMC records:',len(records),'valid fix records:',len(valid),'invalid fix records:',len(records)-len(valid))
        print('  Valid period:',clock(valid[0]['t']),'-',clock(valid[-1]['t']))
        print(format_stats(stats))
        export_csv(records,args.out_dir/f'test{i}_decoded.csv')
    # Diagnostic precision of Test 1: all valid GPS fixes and GPS 3D fixes only.
    all_s,valid1=metrics(all_valid[0]); s3,valid_3d=metrics(all_valid[0], lambda r:r.get('gsa_fix_dimension')==3)
    print('\nTEST 1: 3D FIX ONLY')
    print(format_stats(s3))
    for threshold in [60,90,120,150,180]:
        cutoff=valid1[0]['t']+threshold
        stats,part=metrics(valid1,lambda r:r['t']>=cutoff)
        print(f'\nTEST 1: after {threshold}s of valid fixes ({len(part)} fixes)')
        print('  R95',fmt(stats.get('radial_p95_m',math.nan)),'radial RMS',fmt(stats.get('radial_rms_m',math.nan)),'mean speed km/h',fmt(stats.get('mean_speed_kmh',math.nan)), 'max speed',fmt(stats.get('max_speed_kmh',math.nan)))
    plot_trajectories(all_valid,args.out_dir)
    plot_speeds(all_valid,args.out_dir)
    last_stable = [r for r in valid1 if r['t'] >= valid1[0]['t'] + 120]
    plot_stationary(last_stable,args.out_dir,'Test 1: final 99 s of stable observations')
    print(f'\nSaved plots and parsed CSV to {args.out_dir}')

if __name__=='__main__':
    main()
