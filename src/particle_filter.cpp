/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;


default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	// x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 100;

	normal_distribution<double> noise_x(0, std[0]);
	normal_distribution<double> noise_y(0, std[1]);
	normal_distribution<double> noise_theta(0, std[2]);

	
	for (int i = 0; i < num_particles; ++i) {
		Particle p;
		p.id = i;
		p.x = x + noise_x(gen);
		p.y = y + noise_y(gen);
		p.theta = theta + noise_theta(gen);
		p.weight = 1.0;	 
		particles.push_back(p);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	//  TODO: Add measurements to each particle and add random Gaussian noise.
	//  NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	normal_distribution<double> noise_x(0, std_pos[0]);
	normal_distribution<double> noise_y(0, std_pos[1]);
	normal_distribution<double> noise_theta(0, std_pos[2]);

	for (int i = 0; i < num_particles; ++i) {
		if (fabs(yaw_rate) < 1e-6) {  
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
      		particles[i].y += velocity * delta_t * sin(particles[i].theta);
		} else {
			particles[i].x += velocity/yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
			particles[i].y += velocity/yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}
		particles[i].x += noise_x(gen);
		particles[i].y += noise_y(gen);
		particles[i].theta += noise_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (int i = 0; i < observations.size(); i++) {

		LandmarkObs obs = observations[i];
		double min_dist = numeric_limits<double>::max();
		int min_idx = -1;
		for (int j = 0; j < predicted.size(); j++) {
			LandmarkObs pred = predicted[j];

			double distance = dist(obs.x, obs.y, pred.x, pred.y);

			if (distance < min_dist) {
				min_dist  = distance;
				min_idx = pred.id;
			}
		}

		observations[i].id = min_idx;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
  	for (int i = 0; i < num_particles; i++) {
    	double p_x = particles[i].x;
    	double p_y = particles[i].y;
    	double p_theta = particles[i].theta;

    	// Get Observations transformed coordinates
    	vector<LandmarkObs> observations_transformed;
    	for (int j = 0; j < observations.size(); j++) {
    		double x = observations[j].x * cos(p_theta) - observations[j].y * sin(p_theta) + p_x;
    		double y = observations[j].x * sin(p_theta) + observations[j].y * cos(p_theta) + p_y;
    		observations_transformed.push_back(LandmarkObs{observations[j].id, x, y});
    	}

    	// Predictions within sensor range
    	vector<LandmarkObs> predicted;
    	for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      		float landmark_x = map_landmarks.landmark_list[j].x_f;
      		float landmark_y = map_landmarks.landmark_list[j].y_f;
      		int landmark_id = map_landmarks.landmark_list[j].id_i;
    		// if (dist(landmark_x, landmark_y, p_x, p_y) < sensor_range) {
    		if (fabs(landmark_x - p_x) <= sensor_range && fabs(landmark_y - p_y) <= sensor_range) {
    			predicted.push_back(LandmarkObs{landmark_id, landmark_x, landmark_y});
    		} 
    	}

    	// Get data associations
    	dataAssociation(predicted, observations_transformed);

    	particles[i].weight = 1.0;

    	for (int j=0; j< observations_transformed.size(); j++) {
    		double mu_x, mu_y, x, y, std_x, std_y;
    		x = observations_transformed[j].x;
    		y = observations_transformed[j].y;
    		std_x = std_landmark[0];
    		std_y = std_landmark[1];

    		// TODO: Hash data association to avoid extra for loop
    		for (int k = 0; k < predicted.size(); k++) {
       			if (predicted[k].id == observations_transformed[j].id) {
          			mu_x = predicted[k].x;
          			mu_y = predicted[k].y;
        		}
      		}

      		particles[i].weight *= (1/(2*M_PI*std_x*std_y))*exp(- (pow(x-mu_x, 2) / (2*pow(std_x, 2)) ) - ( pow(y-mu_y, 2)/(2*pow(std_y, 2)) ) );;
    	}
  	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	vector<Particle> new_particles;

  	// get all of the current weights
  	vector<double> probs;
  	for (int i = 0; i < num_particles; i++) {
    	probs.push_back(particles[i].weight);
  	}

  	discrete_distribution<> dist(probs.begin(), probs.end());
  	
  	// Resample
  	for (int i = 0; i < num_particles; i++) {
    	new_particles.push_back(particles[dist(gen)]);
  	}

  	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y) {
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
