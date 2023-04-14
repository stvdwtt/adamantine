/* Copyright (c) 2023, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef MG2_HEAT_SOURCE_HH
#define MG2_HEAT_SOURCE_HH

#include <HeatSource.hh>

namespace adamantine
{
/**
 * A derived class from HeatSource for the modified Gaussian model of a laser
 * heat source that is used by John Coleman, Gerry Knapp, et al. This class is a
 * specialization of the k=2 case. This is almost identical to the Goldak case,
 * but the laser dimensions are scaled differently.
 */
template <int dim>
class MG2HeatSource final : public HeatSource<dim>
{
public:
  /**
   * Constructor.
   * \param[in] database requires the following entries:
   *   - <B>absorption_efficiency</B>: double in \f$[0,1]\f$
   *   - <B>depth</B>: double in \f$[0,\infty)\f$
   *   - <B>diameter</B>: double in \f$[0,\infty)\f$
   *   - <B>max_power</B>: double in \f$[0, \infty)\f$
   *   - <B>input_file</B>: name of the file that contains the scan path
   *     segments
   */
  MG2HeatSource(boost::property_tree::ptree const &database);

  /**
   * Set the time variable.
   */
  void update_time(double time) final;

  /**
   * Returns the value of a MG2 heat source at a specified point and
   * time.
   */
  double value(dealii::Point<dim> const &point,
               double const height) const final;

private:
  dealii::Point<3> _beam_center;
  double _alpha;
  double const _pi_to_1p5_times_0p5 = 0.5 * std::pow(dealii::numbers::PI, 1.5);
};
} // namespace adamantine

#endif
