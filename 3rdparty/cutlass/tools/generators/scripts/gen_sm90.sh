python gen_sm90_operators.py -o mma_sm90_gmma.hpp
python gen_sm90_operators.py -o mma_sm90_gmma_sparse.hpp -s
python gen_sm90_operators.py -o mma_sm90_gmma_ext.hpp -e
python gen_sm90_operators.py -o mma_sm90_gmma_sparse_ext.hpp -s -e
python gen_sm90_traits.py -o mma_traits_sm90_gmma.hpp
python gen_sm90_traits.py -o mma_traits_sm90_gmma_sparse.hpp -s
python gen_sm90_traits.py -o mma_traits_sm90_gmma_ext.hpp -e
python gen_sm90_traits.py -o mma_traits_sm90_gmma_sparse_ext.hpp -e -s
python gen_sm90_selectors.py -o mma_sm90.hpp
