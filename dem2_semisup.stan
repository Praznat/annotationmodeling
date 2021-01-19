
data {
    // observed data
    int<lower=0> NDATA;
    int<lower=0> NITEMS;
    int<lower=0> NUSERS;
    int<lower=1, upper=NITEMS> items[NDATA];
    int<lower=1, upper=NUSERS> u1s[NDATA];
    int<lower=1, upper=NUSERS> u2s[NDATA];
    int<lower=0, upper=NUSERS> n_gold_users; // first n_gold_users assumed to be gold
    real<lower=0> gold_uerr[NUSERS];
    real distances[NDATA];

    // hyperparameters
    real<lower=0> uerr_prior_scale;
    real<lower=0> diff_prior_scale;

    real<lower=0> uerr_prior_loc_scale;
    real<lower=0> diff_prior_loc_scale;
}
parameters {
    real<lower=0> uerr_prior_loc;
    // real<lower=0> diff_prior_loc;
    // real<lower=0> uerr_prior_scale;
    // real<lower=0> diff_prior_scale;
    vector<lower=0>[NUSERS] uerr;
    vector<lower=0>[NITEMS] diff;
}
transformed parameters {
    matrix[NITEMS, NUSERS] label_logprobs = rep_matrix(0, NITEMS, NUSERS);
    matrix[NITEMS, NUSERS] label_probabilities = rep_matrix(0, NITEMS, NUSERS);
    vector[NITEMS] item_logprobs = rep_vector(0, NITEMS);
    vector<lower=0>[NUSERS] uerr_g = uerr;
    if (n_gold_users > 0) {
        for (u in 1:NUSERS) {
            uerr_g[u] = gold_uerr[u];
        }
    }

    for (n in 1:NDATA) {
        int i = items[n];
        int u1 = u1s[n];
        int u2 = u2s[n];
        real dist = distances[n];
        real scale = (uerr_g[u1] + uerr_g[u2]) * diff[i] + 0.01;

        label_logprobs[i, u1] += normal_lpdf(dist | 0, scale);
        label_logprobs[i, u2] += normal_lpdf(dist | 0, scale);
    }
    {
        int n_itemlabels[NITEMS];
        int active_users[NITEMS, NUSERS];
        for (i in 1:NITEMS) {
            int nlabels = 0;
            for (u in 1:NUSERS) {
                if (label_logprobs[i, u] != 0) {
                    nlabels += 1;
                    active_users[i, nlabels] = u;
                }
            }
            n_itemlabels[i] = nlabels;
            if (nlabels > 0) {
                row_vector[nlabels] label_logprobs_v = label_logprobs[i, active_users[i, 1:nlabels]];
                vector[nlabels] probs = softmax(label_logprobs_v');
                for (n in 1:nlabels) {
                    int u = active_users[i, n];
                    label_probabilities[i, u] = probs[n];
                }
                item_logprobs[i] = log_sum_exp(label_logprobs_v);
            }
        }
    }
}
model {
    for (u in 1:NUSERS) {
        uerr[u] ~ normal(uerr_prior_loc, uerr_prior_scale);
    }
    // uerr ~ normal(uerr_prior_loc, uerr_prior_scale);
    // diff ~ normal(diff_prior_loc, diff_prior_scale);
    // uerr_prior_loc ~ normal(0, uerr_prior_loc_scale);
    // diff_prior_loc ~ normal(0, diff_prior_loc_scale);

    for (i in 1:NITEMS) {
        target += item_logprobs[i];
    }
}
generated quantities {
    matrix<lower=0>[NITEMS, NUSERS] dist_from_truth = 1 - label_probabilities;
}
