"""Read-only checks of deployed artifacts and configured Supabase API tables."""
import argparse
import asyncio
import json
import httpx
from fastapi import HTTPException
from app.config import Settings
from app.supabase import SupabaseGateway
from ml.artifacts import read_bundle


async def check_remote(settings):
    results={}
    async with httpx.AsyncClient(timeout=20) as client:
        gateway=SupabaseGateway(settings,client)
        for table in ('persons','reference_images','matches','profiles'):
            try:
                rows=await gateway.list(table,'id',limit='1')
                results[table]={'reachable':isinstance(rows,list),'has_records':bool(rows)}
            except HTTPException as error:
                results[table]={'reachable':False,'status':error.status_code,'message':error.detail}
    return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--remote',action='store_true')
    args=parser.parse_args()
    settings=Settings()
    bundle=read_bundle(settings.models/'pipeline.pkl')
    report={'artifacts_valid':True,'detector':bundle['detector']['architecture'],
            'local_training':bundle['detector']['local_training'],
            'calibration':bundle['calibration']['status'],'disaster_validated':bundle['deployment']['disaster_validated'],
            'database_configured':settings.configured}
    if args.remote:
        report['database']=asyncio.run(check_remote(settings))
    print(json.dumps(report,indent=2))
    if args.remote and not all(t['reachable'] for t in report['database'].values()):
        raise SystemExit(1)


if __name__=='__main__':
    main()
