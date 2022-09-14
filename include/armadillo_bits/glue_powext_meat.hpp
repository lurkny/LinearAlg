// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2008-2016 Conrad Sanderson (http://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------



//! \addtogroup glue_powext
//! @{


template<typename T1, typename T2>
inline
void
glue_powext::apply(Mat<typename T1::elem_type>& out, const Glue<T1, T2, glue_powext>& X)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const quasi_unwrap<T1> UA(X.A);
  const quasi_unwrap<T2> UB(X.B);
  
  const Mat<eT>& A = UA.M;
  const Mat<eT>& B = UB.M;
  
  arma_debug_assert_same_size(A, B, "element-wise pow()");
  
  // TODO: investigate use of Proxy for X.A and X.B
  // TODO: investigate use of openmp
  
  const bool UA_bad_alias = UA.is_alias(out) && (UA.has_subview);  // allow inplace operation
  const bool UB_bad_alias = UB.is_alias(out);
  
  if(UA_bad_alias || UB_bad_alias)
    {
    Mat<eT> tmp;
    
    glue_powext::apply(tmp, A, B);
    
    out.steal_mem(tmp);
    }
  else
    {
    glue_powext::apply(out, A, B);
    }
  }



template<typename eT>
inline
void
glue_powext::apply(Mat<eT>& out, const Mat<eT>& A, const Mat<eT>& B)
  {
  arma_extra_debug_sigprint();
  
  out.set_size(A.n_rows, A.n_cols);
  
  const uword N = out.n_elem;
  
        eT* out_mem = out.memptr();
  const eT*   A_mem =   A.memptr();
  const eT*   B_mem =   B.memptr();
  
  for(uword i=0; i<N; ++i)
    {
    out_mem[i] = eop_aux::pow(A_mem[i], B_mem[i]);
    }
  }



//! @}
