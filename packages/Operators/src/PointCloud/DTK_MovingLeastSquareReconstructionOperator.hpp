//---------------------------------------------------------------------------//
/*
  Copyright (c) 2012, Stuart R. Slattery
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  *: Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  *: Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  *: Neither the name of the University of Wisconsin - Madison nor the
  names of its contributors may be used to endorse or promote products
  derived from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
//---------------------------------------------------------------------------//
/*!
 * \file   DTK_MovingLeastSquareReconstructionOperator.hpp
 * \author Stuart R. Slattery
 * \brief Parallel moving least square interpolator.
 */
//---------------------------------------------------------------------------//

#ifndef DTK_MOVINGLEASTSQUARERECONSTRUCTIONOPERATOR_HPP
#define DTK_MOVINGLEASTSQUARERECONSTRUCTIONOPERATOR_HPP

// added by QC
// define a symbol that enable writing indicator values to files
#define TUNING_INDICATOR_VALUES

#ifdef TUNING_INDICATOR_VALUES
#include <string>
#endif
// added QC

#include "DTK_MapOperator.hpp"
#include "DTK_RadialBasisPolicy.hpp"

#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_RCP.hpp>

#include <Teuchos_SerialDenseMatrix.hpp> // added, QC

#include <Tpetra_CrsMatrix.hpp>

namespace DataTransferKit
{

// added, QC
// forward pairing
template <int DIM>
class SplineInterpolationPairing;
// forward distributor
template <int DIM>
class CenterDistributor;
// added, QC

//---------------------------------------------------------------------------//
/*!
 * \class MovingLeastSquareReconstructionOperator
 * \brief Parallel moving least square interpolator MapOperator
 * implementation.
 */
//---------------------------------------------------------------------------//
template <class Basis, int DIM>
class MovingLeastSquareReconstructionOperator : virtual public MapOperator
{
  public:
    //@{
    //! Typedefs.
    typedef MapOperator Base;
    typedef typename Base::Root Root;
    typedef typename Root::scalar_type Scalar;
    typedef typename Root::local_ordinal_type LO;
    typedef typename Root::global_ordinal_type GO;
    typedef typename Base::TpetraMultiVector TpetraMultiVector;
    typedef typename Base::TpetraMap TpetraMap;
    typedef RadialBasisPolicy<Basis> BP;
    //@}

    /*!
     * \brief Constructor.
     *
     * \param domain_map Parallel map for domain vectors this map should be
     * compatible with.
     *
     * \param range_map Parallel map for range vectors this map should be
     * compatible with.
     */
    MovingLeastSquareReconstructionOperator(
        const Teuchos::RCP<const TpetraMap> &domain_map,
        const Teuchos::RCP<const TpetraMap> &range_map,
        const Teuchos::ParameterList &parameters );

  protected:
    /*
     * \brief Setup the map operator from a domain entity set and a range
     * entity set.
     *
     * \param domain_function The function that contains the data that will be
     * sent to the range. Must always be nonnull but the pointers it contains
     * may be null of no entities are on-process.
     *
     * \param range_space The function that will receive the data from the
     * domain. Must always be nonnull but the pointers it contains to entity
     * data may be null of no entities are on-process.
     *
     * \param parameters Parameters for the setup.
     */
    void setupImpl( const Teuchos::RCP<FunctionSpace> &domain_space,
                    const Teuchos::RCP<FunctionSpace> &range_space ) override;

    /*!
     * \brief Apply the operator.
     */
    // NOTE we use mode == TRANS indicate doing post disc correction, QC
    // NOTE we use alpha to pass in the smoothness indicator threshold sigma
    void applyImpl(
        const TpetraMultiVector &X, TpetraMultiVector &Y,
        Teuchos::ETransp mode = Teuchos::NO_TRANS,
        double alpha = Teuchos::ScalarTraits<double>::one(),
        double beta = Teuchos::ScalarTraits<double>::zero() ) const override;

    /*
     * \brief Transpose apply option.
     */
    bool hasTransposeApplyImpl() const override;

    // added QC for post-processing

    /*!
     * \brief send the source values to the target decomposition map
     *
     * \param[in] domainV domain values
     * \param[out] domainDistV distributed domain/source values
     *
     */
    void sendSource2TargetMap(
        const TpetraMultiVector &domainV,
        Teuchos::SerialDenseMatrix<LO, double> &domainDistV ) const;

    /*!
     * \brief detect and resolve discontinuous solutions
     *
     * \param[in] domainDistV the distributed source values
     * \param[in,out] rangeIntrmV target intermediate solution after
     * transferring
     * \param[in] sigma threshold for smoothness indicator
     * \return Number of disc points
     *
     * We first detect the disc regions with a smooth indicator, similar to
     * those used in WENO schemes. Then a simple cropping strategy is used to
     * ensure the values are bounded locally in the stencil.
     */
    LO detectResolveDisc(
        const Teuchos::SerialDenseMatrix<LO, double> &domainDistV,
        TpetraMultiVector &rangeIntrmV, const double sigma ) const;

    // added QC

  private:
    // Extract node coordinates and ids from an iterator.
    void getNodeCoordsAndIds( const Teuchos::RCP<FunctionSpace> &space,
                              const int entity_dim,
                              Teuchos::ArrayRCP<double> &centers,
                              Teuchos::ArrayRCP<GO> &support_ids ) const;

  private:
    // Flag for search type. True if kNN, false if radius.
    bool d_use_knn;

    // k-nearest-neighbors for support.
    int d_knn;

    // Basis radius for support.
    double d_radius;

    // Domain entity topological dimension. Default is 0 (vertex).
    int d_domain_entity_dim;

    // Range entity topological dimension. Default is 0 (vertex).
    int d_range_entity_dim;

    // leaf size for the kdtree
    // added by QC
    int d_leaf;

    // use qrcp impl
    // added by QC
    bool d_use_qrcp;

    // store the pairing
    // added by QC
    Teuchos::RCP<SplineInterpolationPairing<DIM>> d_pairings;

    // store the distributor
    // added by QC
    Teuchos::RCP<CenterDistributor<DIM>> d_dist;

    // flag for post-processing correction
    // added by QC
    bool d_do_post;

    // local problem row scaling rho, i.e. rows=rho*num_col
    // added by QC
    double d_rho;

    // save the point clouds of target and source for post-processing
    // the source is distributed point cloud
    // added by QC
    // Teuchos::ArrayRCP<double> d_tar_pts;
    // Teuchos::Array<double> d_src_pts;

    // added a reference to the parameter
    // added by QC
    Teuchos::ParameterList &d_pars;

    // Coupling matrix.
    Teuchos::RCP<Tpetra::CrsMatrix<Scalar, LO, GO>> d_coupling_matrix;

// added QC
#ifdef TUNING_INDICATOR_VALUES
    std::string d_file_name;
#endif
    // added QC
};

//---------------------------------------------------------------------------//

} // end namespace DataTransferKit

//---------------------------------------------------------------------------//

#endif // end DTK_MOVINGLEASTSQUARERECONSTRUCTIONOPERATOR_HPP

//---------------------------------------------------------------------------//
// end DTK_MovingLeastSquareReconstructionOperator.hpp
//---------------------------------------------------------------------------//
