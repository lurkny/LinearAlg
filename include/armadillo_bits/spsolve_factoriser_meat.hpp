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


//! \addtogroup spsolve_factoriser
//! @{



template<typename worker_ptr_type>
inline
void
spsolve_factoriser::delete_worker()
  {
  arma_extra_debug_sigprint();
  
  if(worker_ptr != nullptr)
    {
    worker_ptr_type ptr = reinterpret_cast<worker_ptr_type>(worker_ptr);
    
    delete ptr;
    
    worker_ptr = nullptr;
    }
  }



inline
void
spsolve_factoriser::cleanup()
  {
  arma_extra_debug_sigprint();
  
  #if defined(ARMA_USE_SUPERLU)
    {
         if(elem_type_indicator == 1)  { delete_worker< superlu_worker<    float>* >(); }
    else if(elem_type_indicator == 2)  { delete_worker< superlu_worker<   double>* >(); }
    else if(elem_type_indicator == 3)  { delete_worker< superlu_worker< cx_float>* >(); }
    else if(elem_type_indicator == 4)  { delete_worker< superlu_worker<cx_double>* >(); }
    
    elem_type_indicator = 0;
    rcond_value         = double(0);
    }
  #endif
  }



inline
spsolve_factoriser::~spsolve_factoriser()
  {
  arma_extra_debug_sigprint_this(this);
  
  cleanup();
  }



inline
spsolve_factoriser::spsolve_factoriser()
  {
  arma_extra_debug_sigprint_this(this);
  }



inline
double
spsolve_factoriser::rcond() const
  {
  arma_extra_debug_sigprint();
  
  return rcond_value;
  }



template<typename eT>
inline
bool
spsolve_factoriser::factorise(const SpMat<eT>& A)
  {
  arma_extra_debug_sigprint();
  
  // TODO: allow user options
  
  #if defined(ARMA_USE_SUPERLU)
    {
    typedef typename get_pod_type<eT>::result T;
    
    typedef superlu_worker<eT>  worker_type;
    typedef superlu_worker<eT>* worker_ptr_type;
    
    cleanup();
    
    worker_ptr = new(std::nothrow) worker_type;
    
    if(worker_ptr == nullptr)
      {
      arma_debug_warn_level(3, "spsolve_factoriser::factorise(): could not construct worker object");
      return false;
      }
    
         if(    is_float<eT>::value)  { elem_type_indicator = 1; }
    else if(   is_double<eT>::value)  { elem_type_indicator = 2; }
    else if( is_cx_float<eT>::value)  { elem_type_indicator = 3; }
    else if(is_cx_double<eT>::value)  { elem_type_indicator = 4; }
    
    worker_ptr_type local_worker_ptr = reinterpret_cast<worker_ptr_type>(worker_ptr);
    worker_type&    local_worker_ref = (*local_worker_ptr);
    
    T local_rcond_value = T(0);
    
    const bool status = local_worker_ref.factorise(local_rcond_value, A);
    
    rcond_value = double(local_rcond_value);
    
    if(status == false)
      {
      arma_debug_warn_level(3, "spsolve_factoriser::factorise(): factorisation failed");
      delete_worker<worker_ptr_type>();
      return false;
      }
    
    if(local_rcond_value <= std::numeric_limits<eT>::epsilon())
      {
      arma_debug_warn_level(2, "spsolve_factoriser::factorise(): system is singular to working precision; rcond: ", local_rcond_value);
      }
    
    return true;
    }
  #else
    {
    arma_ignore(A);
    arma_stop_logic_error("spsolve_factoriser::factorise(): use of SuperLU must be enabled");
    return false;
    }
  #endif
  }



template<typename eT>
inline
bool
spsolve_factoriser::solve(Mat<eT>& X, const Mat<eT>& B)
  {
  arma_extra_debug_sigprint();
  
  #if defined(ARMA_USE_SUPERLU)
    {
    typedef superlu_worker<eT>  worker_type;
    typedef superlu_worker<eT>* worker_ptr_type;
    
    if(worker_ptr == nullptr)
      {
      arma_debug_warn_level(2, "spsolve_factoriser::solve(): no factorisation available");
      X.reset();
      return false;
      }
    
    bool type_mismatch = false;
    
         if(    (is_float<eT>::value) && (elem_type_indicator != 1) )  { type_mismatch = true; }
    else if(   (is_double<eT>::value) && (elem_type_indicator != 2) )  { type_mismatch = true; }
    else if( (is_cx_float<eT>::value) && (elem_type_indicator != 3) )  { type_mismatch = true; }
    else if((is_cx_double<eT>::value) && (elem_type_indicator != 4) )  { type_mismatch = true; }
    
    if(type_mismatch)
      {
      arma_debug_warn_level(1, "spsolve_factoriser::solve(): matrix type mismatch");
      X.reset();
      return false;
      }
    
    worker_ptr_type local_worker_ptr = reinterpret_cast<worker_ptr_type>(worker_ptr);
    worker_type&    local_worker_ref = (*local_worker_ptr);
    
    const bool status = local_worker_ref.solve(X,B);
    
    if(status == false)
      {
      arma_debug_warn_level(3, "spsolve_factoriser::solve(): solution not found");
      X.reset();
      return false;
      }
    
    return true;
    }
  #else
    {
    arma_ignore(X);
    arma_ignore(B);
    arma_stop_logic_error("spsolve_factoriser::solve(): use of SuperLU must be enabled");
    return false;
    }
  #endif
  }



//! @}
