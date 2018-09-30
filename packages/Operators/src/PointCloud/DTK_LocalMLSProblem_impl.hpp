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
 * \file   DTK_LocalMLSProblem_impl.hpp
 * \author Stuart R. Slattery
 * \brief  Local moving least square problem.
 */
//---------------------------------------------------------------------------//

#ifndef DTK_LOCALMLSPROBLEM_IMPL_HPP
#define DTK_LOCALMLSPROBLEM_IMPL_HPP

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>

#include "DTK_DBC.hpp"
#include "DTK_EuclideanDistance.hpp"
#include "DTK_LocalMLSProblem.hpp"
#include "DTK_RadialBasisPolicy.hpp"

#include <Teuchos_LAPACK.hpp>
#include <Teuchos_SerialDenseSolver.hpp>
#include <Teuchos_SerialDenseVector.hpp>

// added, QC
// expand the locally adaptive radius by the following percentage
#define EXPANDED_PERCENTAGE 0.1

#if defined( INTEL_CXML )
#define PREFIX __stdcall
#define Teuchos_fcd const char *, unsigned int
#elif defined( INTEL_MKL )
#define PREFIX
#define Teuchos_fcd const char *
#else /* Not CRAY_T3X or INTEL_CXML or INTEL_MKL */
#define PREFIX
#define Teuchos_fcd const char *
#endif

// add the condition number estimator for triangle systems
#define DTRCON F77_BLAS_MANGLE( dtrcon, DTRCON )
extern "C"
{
    extern void PREFIX DTRCON( Teuchos_fcd, Teuchos_fcd, Teuchos_fcd, int *,
                               double *, int *, double *, double *, int *,
                               int * );
}
// added, QC

namespace DataTransferKit
{

// added, QC
// build weighted Vandermonde system with row weights and column scaling
template <int DIM>
struct WeightedVandermonde
{
    typedef Teuchos::SerialDenseVector<int, double> map_t;
    inline void build( const Teuchos::ArrayView<const double> &target_center,
                       const Teuchos::ArrayView<const unsigned> &source_lids,
                       const Teuchos::ArrayView<const double> &source_centers,
                       const int num_sources,
                       const Teuchos::SerialDenseVector<int, double> &w,
                       const double s,
                       Teuchos::SerialDenseMatrix<int, double> &V );
};
// added, QC

//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
template <class Basis, int DIM>
LocalMLSProblem<Basis, DIM>::LocalMLSProblem(
    const Teuchos::ArrayView<const double> &target_center,
    const Teuchos::ArrayView<const unsigned> &source_lids,
    const Teuchos::ArrayView<const double> &source_centers, const Basis &basis,
    const double radius_, bool use_new_impl, const double rho )
    : d_shape_function( source_lids.size() )
{
    DTK_REQUIRE( 0 == source_centers.size() % DIM );
    DTK_REQUIRE( 0 == target_center.size() % DIM );

    // Number of source centers supporting this target center.
    int num_sources = source_lids.size();

    // get the preferred row size
    // added QC
    // for each dimension, a preferred choice of row size is 1.5*column
    // DTK2 only uses quadratic polynomial basis thus having the following table
    const static int COLS[3] = {3, 6, 10};

    // However, for solution transfer problem, we have observed that a larger
    // neighborhood can super-converge. Therefore, we choose 3*column, i.e.
    const static int PREFERRED_ROWS[3] = {9, 18, 30};

    // we set the 1.5*col to be the min requirement, and passin rho to tune
    // the choice of rows, i.e. number of points
    const static int MIN_ROWS[3] = {5, 9, 15};

    int row_choice = PREFERRED_ROWS[DIM - 1];
    if ( rho > 0.0 && rho * COLS[DIM - 1] >= MIN_ROWS[DIM - 1] )
        row_choice = std::ceil( rho * COLS[DIM - 1] );
    if ( num_sources > row_choice )
    {
        num_sources = row_choice;
    }
    d_shape_function.resize( num_sources, 0.0 );
    // use adaptive radius
    double radius = radius_; // remove unused warning
    // since the source centers are sorted, getting the radius of the local
    // support region is trivial
    {
        // implement in the local scope by copying the following code
        Teuchos::ArrayView<const double> source_center_view;
        source_center_view =
            source_centers( DIM * source_lids[num_sources - 1], DIM );
        radius = EuclideanDistance<DIM>::distance(
            target_center.getRawPtr(), source_center_view.getRawPtr() );
        d_radii = radius;
        radius *= ( 1.0 + EXPANDED_PERCENTAGE ); // expand the radius
    }
    // added QC

    // added, QC
    if ( use_new_impl )
    {
        // the new implementation

        const int num_cols = COLS[DIM - 1];

        Teuchos::ArrayView<const double> source_center_view;

        // create row weights array
        Teuchos::SerialDenseVector<int, double> w( num_sources );
        // create column geometrical scaling factor
        double s = 0.0;
        // build row weights and geometrical scaling factor
        // NOTE that we cannot use the point with largest radius for the
        // scaling factor, since it may not have the largest value in
        // each of the dimension
        for ( int i = 0; i < num_sources; ++i )
        {
            source_center_view = source_centers( DIM * source_lids[i], DIM );
            for ( int dim = 0; dim < DIM; ++dim )
                s = std::max( s, std::abs( source_center_view[dim] -
                                           target_center[dim] ) );
            // compute row weights
            const double dist = EuclideanDistance<DIM>::distance(
                target_center.getRawPtr(), source_center_view.getRawPtr() );
            w[i] = BP::evaluateValue( basis, radius, dist );
        }

        if ( s == 0.0 )
            s = 1.0;
        s = 1.0 / s; //  inverse s

        // create Vandermonde system storage
        Teuchos::SerialDenseMatrix<int, double> V( num_sources, num_cols );
        // build Vandermonde system
        WeightedVandermonde<DIM>().build(
            target_center, source_lids, source_centers, num_sources, w, s, V );

        // solve the system with TQRCP
        Teuchos::LAPACK<int, double> lapack;
        int info = 0;
        Teuchos::Array<double> tau( std::min( num_sources, num_cols ) );
        Teuchos::Array<double> work( 3 * num_sources );
        Teuchos::Array<int> jpvt( num_cols, 0 );
        // query the optimal work size
        lapack.GEQP3( num_sources, num_cols, V.values(), num_sources,
                      jpvt.getRawPtr(), tau.getRawPtr(), work.getRawPtr(), -1,
                      nullptr, &info );
        DTK_CHECK( 0 == info );
        //================================
        // compute the QRCP factorization
        //================================
        work.resize( work[0] );
        lapack.GEQP3( num_sources, num_cols, V.values(), num_sources,
                      jpvt.getRawPtr(), tau.getRawPtr(), work.getRawPtr(),
                      work.size(), nullptr, &info );
        DTK_CHECK( 0 == info );
        // to zero-based
        // for ( auto itr = jpvt.begin(); itr != jpvt.end(); ++itr )
        //     *itr -= 1;
        //================================================
        // truncation step with tri system cond estimator
        //================================================
        // estimate the condition number to get the leading full-rank
        // note that the upper part is triangle system thus calling
        // trcon routine with 1norm and upper-uplo
        int rank = num_cols; // initial rank
        double cond;
        const static double tol = 1e-12; // we might need to increase this
        static char NORM = '1', UPLO = 'U', DIAG = 'N';
        if ( work.size() < 3 * num_cols )
            work.resize( 3 * num_cols );
        Teuchos::Array<int> iwork( num_cols );
        do
        {
            // estimate the reciprocal of condition number
            DTRCON( &NORM, &UPLO, &DIAG, &rank, V.values(), &num_sources, &cond,
                    work.getRawPtr(), iwork.getRawPtr(), &info );
            DTK_CHECK( 0 == info );
            // if cond is good, then break
            if ( cond >= tol )
                break;
            --rank;
            if ( rank <= 0 )
                break;
        } while ( true );
        DTK_CHECK( rank != 0 );
#ifndef NDEBUG
        if ( rank < num_cols )
            std::cerr << "\nWARNING! rank truncated in QRCP from " << num_cols
                      << " to " << rank << ", " << __FILE__ << ':' << __LINE__
                      << '\n'
                      << std::endl;
#endif

        //=======================================
        // form the coefficient by implicit diff
        //=======================================

        // now we solve it implicitly, Notice that the column scaling
        // matrix for the first column is 1 (implicitly given) thus no special
        // treatment is needed for the rhs
        //
        // solve for
        //      S*V'*W*inv(W)*c=S*e_1=e_1
        // let VV = S*V'*W, and cc = W^{-1}*c, VV is what we have formed
        //  =>  VV'*cc=e_1
        // The QRCP is to decompose VV into VV*P=Q*R, thus
        //  =>  P*R'*Q'*cc=e_1
        //  =>  R'*Q'*cc=P'*e_1=b
        // where b is rhs, P' is the row permutation matrix,
        // P is column permutation though
        //  =>  cc = Q*inv(R')*b
        //  =>  c = W*cc
        for ( int i = 0; i < rank; ++i )
            if ( jpvt[i] == 1 ) // one based pivoting array
                d_shape_function[i] = 1.0;

        // transpose tri solve
        lapack.TRTRS( UPLO, 'T', DIAG, rank, 1, V.values(), num_sources,
                      d_shape_function.getRawPtr(), num_sources, &info );
        DTK_CHECK( info == 0 );

        // form Q*rhs
        // first query the size
        lapack.ORMQR( 'L', 'N', num_sources, 1, rank, V.values(), num_sources,
                      tau.getRawPtr(), d_shape_function.getRawPtr(),
                      num_sources, work.getRawPtr(), -1, &info );
        DTK_CHECK( info == 0 );
        work.resize( work[0] );
        // then compute the implicit MV
        lapack.ORMQR( 'L', 'N', num_sources, 1, rank, V.values(), num_sources,
                      tau.getRawPtr(), d_shape_function.getRawPtr(),
                      num_sources, work.getRawPtr(), work.size(), &info );
        DTK_CHECK( info == 0 );

        // finally, clean up everything with multiplying the row weights
        Teuchos::SerialDenseVector<int, double>(
            Teuchos::View, d_shape_function.getRawPtr(), num_sources )
            .scale( w );
    }
    // added, QC
    else
    {
        // the original implementation

        int poly_size = 0;
        Teuchos::SerialDenseMatrix<int, double> P;
        Teuchos::SerialDenseVector<int, double> target_poly;

        // Make Phi.
        Teuchos::SerialDenseMatrix<int, double> phi( num_sources, num_sources );
        Teuchos::ArrayView<const double> source_center_view;
        double dist = 0.0;
        for ( int i = 0; i < num_sources; ++i )
        {
            source_center_view = source_centers( DIM * source_lids[i], DIM );
            dist = EuclideanDistance<DIM>::distance(
                target_center.getRawPtr(), source_center_view.getRawPtr() );
            phi( i, i ) = BP::evaluateValue( basis, radius, dist );
        }

        // Make P.
        Teuchos::Array<int> poly_ids( 1, 0 );
        int poly_id = 0;
        P.reshape( num_sources, poly_id + 1 );
        for ( int i = 0; i < num_sources; ++i )
        {
            source_center_view = source_centers( DIM * source_lids[i], DIM );
            P( i, poly_id ) = polynomialCoefficient( 0, source_center_view );
        }
        ++poly_id;

        // Add polynomial columns until we are full rank.
        bool full_rank = false;
        int num_poly = 10;
        int total_poly = std::min( num_poly, num_sources );
        for ( int j = 1; j < total_poly; ++j )
        {
            // Add the next column.
            P.reshape( num_sources, poly_id + 1 );
            for ( int i = 0; i < num_sources; ++i )
            {
                source_center_view =
                    source_centers( DIM * source_lids[i], DIM );
                P( i, poly_id ) =
                    polynomialCoefficient( j, source_center_view );
            }

            // Check for rank deficiency.
            full_rank = isFullRank( P );

            // If we are full rank, add this coefficient.
            if ( full_rank )
            {
                poly_ids.push_back( j );
                ++poly_id;
            }

            // If we are rank deficient, remove the last column.
            else
            {
                P.reshape( num_sources, poly_id );
            }
        }

        // Make p.
        poly_size = poly_ids.size();
        target_poly.resize( poly_size );
        for ( int i = 0; i < poly_size; ++i )
        {
            target_poly( i ) =
                polynomialCoefficient( poly_ids[i], target_center );
        }

        // Construct b.
        Teuchos::SerialDenseMatrix<int, double> b( poly_size, num_sources );
        b.multiply( Teuchos::TRANS, Teuchos::NO_TRANS, 1.0, P, phi, 0.0 );

        // Construct the A matrix.
        Teuchos::SerialDenseMatrix<int, double> A( poly_size, poly_size );
        {
            // Build A.
            Teuchos::SerialDenseMatrix<int, double> work( num_sources,
                                                          poly_size );
            work.multiply( Teuchos::NO_TRANS, Teuchos::NO_TRANS, 1.0, phi, P,
                           0.0 );
            A.multiply( Teuchos::TRANS, Teuchos::NO_TRANS, 1.0, P, work, 0.0 );
        }

        // Apply the inverse of the A matrix to b.
        Teuchos::LAPACK<int, double> lapack;
        double A_rcond = std::numeric_limits<double>::epsilon();
        Teuchos::Array<double> work( 4 * A.numCols() );
        Teuchos::SerialDenseVector<int, double> s( poly_size );
        int rank = 0;
        int info = 0;

        // Estimate the reciprocal of the condition number.
        Teuchos::Array<int> ipiv( std::min( A.numRows(), A.numCols() ) );
        Teuchos::SerialDenseMatrix<int, double> LU_A( A );
        lapack.GETRF( LU_A.numRows(), LU_A.numCols(), LU_A.values(),
                      LU_A.numRows(), ipiv.getRawPtr(), &info );
        DTK_CHECK( 0 == info );

        Teuchos::Array<int> iwork( A.numCols() );
        lapack.GECON( '1', LU_A.numCols(), LU_A.values(), LU_A.numRows(),
                      A.normOne(), &A_rcond, work.getRawPtr(),
                      iwork.getRawPtr(), &info );
        DTK_CHECK( 0 == info );

        // Get the optimal work size.
        lapack.GELSS( A.numRows(), A.numCols(), b.numCols(), A.values(),
                      A.numRows(), b.values(), b.numRows(), s.values(), A_rcond,
                      &rank, work.getRawPtr(), -1, &info );
        DTK_CHECK( 0 == info );

        // Apply the inverse of A to b.
        work.resize( work[0] );
        lapack.GELSS( A.numRows(), A.numCols(), b.numCols(), A.values(),
                      A.numRows(), b.values(), b.numRows(), s.values(), A_rcond,
                      &rank, work.getRawPtr(), work.size(), &info );
        DTK_CHECK( 0 == info );

        // Construct the basis.
        Teuchos::SerialDenseMatrix<int, double> shape_matrix(
            Teuchos::View, d_shape_function.getRawPtr(), 1, 1,
            d_shape_function.size() );
        shape_matrix.multiply( Teuchos::TRANS, Teuchos::NO_TRANS, 1.0,
                               target_poly, b, 0.0 );
    }
}

//---------------------------------------------------------------------------//
// Get a polynomial coefficient.
template <class Basis, int DIM>
double LocalMLSProblem<Basis, DIM>::polynomialCoefficient(
    const int coeff, const Teuchos::ArrayView<const double> &center ) const
{
    switch ( coeff )
    {
    // Linear.
    case ( 0 ):
        return 1.0;
        break;
    case ( 1 ):
        return center[0];
        break;
    case ( 2 ):
        return center[1];
        break;
    case ( 3 ):
        return center[2];
        break;

    // Quadratic
    case ( 4 ):
        return center[0] * center[1];
        break;
    case ( 5 ):
        return center[0] * center[2];
        break;
    case ( 6 ):
        return center[1] * center[2];
        break;
    case ( 7 ):
        return center[0] * center[0];
        break;
    case ( 8 ):
        return center[1] * center[1];
        break;
    case ( 9 ):
        return center[2] * center[2];
        break;
    }
    return 0.0;
}

//---------------------------------------------------------------------------//
// Check if a matrix is full rank.
template <class Basis, int DIM>
bool LocalMLSProblem<Basis, DIM>::isFullRank(
    const Teuchos::SerialDenseMatrix<int, double> &matrix ) const
{
    // Copy the matrix.
    Teuchos::SerialDenseMatrix<int, double> A = matrix;

    // Determine the full rank.
    int full_rank = std::min( A.numRows(), A.numCols() );

    // Compute the singular value decomposition.
    Teuchos::LAPACK<int, double> lapack;
    Teuchos::Array<double> S( full_rank );
    Teuchos::SerialDenseMatrix<int, double> U( A.numRows(), A.numRows() );
    Teuchos::SerialDenseMatrix<int, double> VT( A.numCols(), A.numCols() );
    Teuchos::Array<double> work( full_rank );
    Teuchos::Array<double> rwork( full_rank );
    int info = 0;
    lapack.GESVD( 'A', 'A', A.numRows(), A.numCols(), A.values(), A.numRows(),
                  S.getRawPtr(), U.values(), U.numRows(), VT.values(),
                  VT.numRows(), work.getRawPtr(), -1, rwork.getRawPtr(),
                  &info );
    DTK_CHECK( 0 == info );

    work.resize( work[0] );
    rwork.resize( work.size() );
    lapack.GESVD( 'A', 'A', A.numRows(), A.numCols(), A.values(), A.numRows(),
                  S.getRawPtr(), U.values(), U.numRows(), VT.values(),
                  VT.numRows(), work.getRawPtr(), work.size(),
                  rwork.getRawPtr(), &info );
    DTK_CHECK( 0 == info );

    // Check the singular values. If they are greater than delta they count.
    double epsilon = std::numeric_limits<double>::epsilon();
    double delta = S[0] * epsilon;
    int rank = std::count_if( S.begin(), S.end(),
                              [=]( double s ) { return ( s > delta ); } );

    return ( rank == full_rank );
}

// added, QC
template <>
void WeightedVandermonde<1>::build(
    const Teuchos::ArrayView<const double> &target_center,
    const Teuchos::ArrayView<const unsigned> &source_lids,
    const Teuchos::ArrayView<const double> &source_centers,
    const int num_sources, const Teuchos::SerialDenseVector<int, double> &w,
    const double s, Teuchos::SerialDenseMatrix<int, double> &V )
{
    // create view var
    Teuchos::ArrayView<const double> view;

    // set the first column to 1
    map_t( Teuchos::View, V[0], num_sources ) = 1.0;
    // set the second and third column
    for ( int i = 0; i < num_sources; ++i )
    {
        view = source_centers( source_lids[i], 1 );
        const double coeff = s * ( view[0] - target_center[0] );
        V( i, 1 ) = coeff;
        V( i, 2 ) = coeff * coeff;
    }

    // apply row weights
    for ( int j = 0; j < 3; ++j )
    {
        map_t( Teuchos::View, V[j], num_sources ).scale( w );
    }
}

template <>
void WeightedVandermonde<2>::build(
    const Teuchos::ArrayView<const double> &target_center,
    const Teuchos::ArrayView<const unsigned> &source_lids,
    const Teuchos::ArrayView<const double> &source_centers,
    const int num_sources, const Teuchos::SerialDenseVector<int, double> &w,
    const double s, Teuchos::SerialDenseMatrix<int, double> &V )
{
    // create view var
    Teuchos::ArrayView<const double> view;

    // since the degree is known, it's easier (but ugly) to hard-code the
    // implementation column by column

    // 1 x y x^2 xy y^2

    // set the first column to 1
    map_t( Teuchos::View, V[0], num_sources ) = 1.0;
    // set the rest columns
    for ( int i = 0; i < num_sources; ++i )
    {
        view = source_centers( 2 * source_lids[i], 2 );
        const double coeffx = s * ( view[0] - target_center[0] ),
                     coeffy = s * ( view[1] - target_center[1] );
        V( i, 1 ) = coeffx;
        V( i, 2 ) = coeffy;
        V( i, 3 ) = coeffx * coeffx;
        V( i, 4 ) = coeffx * coeffy;
        V( i, 5 ) = coeffy * coeffy;
    }

    // apply row weights
    for ( int j = 0; j < 6; ++j )
    {
        map_t( Teuchos::View, V[j], num_sources ).scale( w );
    }
}

template <>
void WeightedVandermonde<3>::build(
    const Teuchos::ArrayView<const double> &target_center,
    const Teuchos::ArrayView<const unsigned> &source_lids,
    const Teuchos::ArrayView<const double> &source_centers,
    const int num_sources, const Teuchos::SerialDenseVector<int, double> &w,
    const double s, Teuchos::SerialDenseMatrix<int, double> &V )
{
    // create view var
    Teuchos::ArrayView<const double> view;

    // since the degree is known, it's easier (but ugly) to hard-code the
    // implementation column by column

    // 1 x y z x^2 xy xz y^2 yz z^2

    // set the first column to 1
    map_t( Teuchos::View, V[0], num_sources ) = 1.0;
    // set the rest columns
    for ( int i = 0; i < num_sources; ++i )
    {
        view = source_centers( 3 * source_lids[i], 3 );
        const double coeffx = s * ( view[0] - target_center[0] ),
                     coeffy = s * ( view[1] - target_center[1] ),
                     coeffz = s * ( view[2] - target_center[2] );
        V( i, 1 ) = coeffx;
        V( i, 2 ) = coeffy;
        V( i, 3 ) = coeffz;
        V( i, 4 ) = coeffx * coeffx;
        V( i, 5 ) = coeffx * coeffy;
        V( i, 6 ) = coeffx * coeffz;
        V( i, 7 ) = coeffy * coeffy;
        V( i, 8 ) = coeffy * coeffz;
        V( i, 9 ) = coeffz * coeffz;
    }

    // apply row weights
    for ( int j = 0; j < 10; ++j )
    {
        map_t( Teuchos::View, V[j], num_sources ).scale( w );
    }
}
// added, QC

//---------------------------------------------------------------------------//

} // end namespace DataTransferKit

//---------------------------------------------------------------------------//

#endif // end DTK_LOCALMLSPROBLEM_IMPL_HPP

//---------------------------------------------------------------------------//
// end DTK_LocalMLSProblem_impl.hpp
//---------------------------------------------------------------------------//
