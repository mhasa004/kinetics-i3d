#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python extract_i3d_large_video.py --video /mnt/research-6f/mhasan/data/vsd/movies/8Mile.mp4 --flow /mnt/research-6f/mhasan/data/vsd/movies/8Mile.flow.avi --outfile /mnt/research-6f/mhasan/data/vsd/movies/8Mile &
CUDA_VISIBLE_DEVICES=2 python extract_i3d_large_video.py --video /mnt/research-6f/mhasan/data/vsd/movies/Armageddon.mp4 --flow /mnt/research-6f/mhasan/data/vsd/movies/Armageddon.flow.avi --outfile /mnt/research-6f/mhasan/data/vsd/movies/Armageddon &
CUDA_VISIBLE_DEVICES=3 python extract_i3d_large_video.py --video /mnt/research-6f/mhasan/data/vsd/movies/BillyElliot.mp4 --flow /mnt/research-6f/mhasan/data/vsd/movies/BillyElliot.flow.avi --outfile /mnt/research-6f/mhasan/data/vsd/movies/BillyElliot &
CUDA_VISIBLE_DEVICES=1 python extract_i3d_large_video.py --video /mnt/research-6f/mhasan/data/vsd/movies/Braveheart.mp4 --flow /mnt/research-6f/mhasan/data/vsd/movies/Braveheart.flow.avi --outfile /mnt/research-6f/mhasan/data/vsd/movies/Braveheart &
wait
CUDA_VISIBLE_DEVICES=1 python extract_i3d_large_video.py --video /mnt/research-6f/mhasan/data/vsd/movies/DeadPoetsSociety.mp4 --flow /mnt/research-6f/mhasan/data/vsd/movies/DeadPoetsSociety.flow.avi --outfile /mnt/research-6f/mhasan/data/vsd/movies/DeadPoetsSociety &
CUDA_VISIBLE_DEVICES=2 python extract_i3d_large_video.py --video /mnt/research-6f/mhasan/data/vsd/movies/Desperado.mp4 --flow /mnt/research-6f/mhasan/data/vsd/movies/Desperado.flow.avi --outfile /mnt/research-6f/mhasan/data/vsd/movies/Desperado &
CUDA_VISIBLE_DEVICES=3 python extract_i3d_large_video.py --video /mnt/research-6f/mhasan/data/vsd/movies/Eragon.mp4 --flow /mnt/research-6f/mhasan/data/vsd/movies/Eragon.flow.avi --outfile /mnt/research-6f/mhasan/data/vsd/movies/Eragon &
CUDA_VISIBLE_DEVICES=0 python extract_i3d_large_video.py --video /mnt/research-6f/mhasan/data/vsd/movies/FantasticFour1.mp4 --flow /mnt/research-6f/mhasan/data/vsd/movies/FantasticFour1.flow.avi --outfile /mnt/research-6f/mhasan/data/vsd/movies/FantasticFour1 &
wait
CUDA_VISIBLE_DEVICES=1 python extract_i3d_large_video.py --video /mnt/research-6f/mhasan/data/vsd/movies/Fargo.mp4 --flow /mnt/research-6f/mhasan/data/vsd/movies/Fargo.flow.avi --outfile /mnt/research-6f/mhasan/data/vsd/movies/Fargo &
CUDA_VISIBLE_DEVICES=2 python extract_i3d_large_video.py --video /mnt/research-6f/mhasan/data/vsd/movies/FightClub.mp4 --flow /mnt/research-6f/mhasan/data/vsd/movies/FightClub.flow.avi --outfile /mnt/research-6f/mhasan/data/vsd/movies/FightClub &
CUDA_VISIBLE_DEVICES=3 python extract_i3d_large_video.py --video /mnt/research-6f/mhasan/data/vsd/movies/ForrestGump.mp4 --flow /mnt/research-6f/mhasan/data/vsd/movies/ForrestGump.flow.avi --outfile /mnt/research-6f/mhasan/data/vsd/movies/ForrestGump &
CUDA_VISIBLE_DEVICES=0 python extract_i3d_large_video.py --video /mnt/research-6f/mhasan/data/vsd/movies/GhostintheShell.mp4 --flow /mnt/research-6f/mhasan/data/vsd/movies/GhostintheShell.flow.avi --outfile /mnt/research-6f/mhasan/data/vsd/movies/GhostintheShell &
wait
CUDA_VISIBLE_DEVICES=1 python extract_i3d_large_video.py --video /mnt/research-6f/mhasan/data/vsd/movies/HarryPotterandtheOrderofthePhoenix.mp4 --flow /mnt/research-6f/mhasan/data/vsd/movies/HarryPotterandtheOrderofthePhoenix.flow.avi --outfile /mnt/research-6f/mhasan/data/vsd/movies/HarryPotterandtheOrderofthePhoenix &
CUDA_VISIBLE_DEVICES=2 python extract_i3d_large_video.py --video /mnt/research-6f/mhasan/data/vsd/movies/IamLegend.mp4 --flow /mnt/research-6f/mhasan/data/vsd/movies/IamLegend.flow.avi --outfile /mnt/research-6f/mhasan/data/vsd/movies/IamLegend &
CUDA_VISIBLE_DEVICES=3 python extract_i3d_large_video.py --video /mnt/research-6f/mhasan/data/vsd/movies/IndependenceDay.mp4 --flow /mnt/research-6f/mhasan/data/vsd/movies/IndependenceDay.flow.avi --outfile /mnt/research-6f/mhasan/data/vsd/movies/IndependenceDay &
CUDA_VISIBLE_DEVICES=0 python extract_i3d_large_video.py --video /mnt/research-6f/mhasan/data/vsd/movies/Jumanji.mp4 --flow /mnt/research-6f/mhasan/data/vsd/movies/Jumanji.flow.avi --outfile /mnt/research-6f/mhasan/data/vsd/movies/Jumanji &
wait