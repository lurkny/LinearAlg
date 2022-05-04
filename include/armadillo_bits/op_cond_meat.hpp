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


//! \addtogroup op_cond
//! @{



template<typename T1>
inline
typename T1::pod_type
op_cond::apply(const Base<typename T1::elem_type, T1>& X)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  typedef typename T1::pod_type   T;
  
  // TODO: implement speed up for symmetric matrices, similar to op_pinv::apply_sym()
  
  Mat<eT> A(X.get_ref());
  
  Col<T> S;
  
  const bool status = auxlib::svd_dc(S, A);
  
  if(status == false)
    {
    arma_debug_warn_level(3, "cond(): svd failed");
    
    return Datum<T>::nan;
    }
  
  return (S.n_elem > 0) ? T( max(S) / min(S) ) : T(0);
  }



//! @}
