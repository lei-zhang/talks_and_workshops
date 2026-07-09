data {
  int<lower=1> nTrials;               
  int<lower=1,upper=2> choice[nTrials];     
  real<lower=-1, upper=1> reward[nTrials]; 
}

transformed data {
  vector[2] initV;  // initial values for V
  initV = rep_vector(0.0, 2);
}

parameters {
  real<lower=0,upper=1> lr;
  real<lower=0,upper=20> tau;  
}

model {
  vector[2] v; // value
  real pe;     // prediction error
  
  v = initV;

  for (t in 1:nTrials) {        
    // compute action probabilities
    choice[t] ~ categorical_logit( tau * v );
    
    // prediction error 
    pe = reward[t] - v[choice[t]];

    // value updating (learning) 
    v[choice[t]] = v[choice[t]] + lr * pe; 
  }  
}
