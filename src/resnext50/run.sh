for i in {2..2};do
  python train.py --fold $i --nfolds 5 --batch_size 16 --epochs 50 --lr 1e-4 --expansion 0 --workers 16
done


#python full_size_validation.py