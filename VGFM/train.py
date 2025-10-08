# import os, sys, json, math, itertools
import pandas as pd, numpy as np
# import warnings

# from tqdm import tqdm
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import torch

from VGFM.utils import sample, generate_steps, compute_uot_plans, get_batch
from VGFM.losses import MMD_loss, OT_loss1, OT_loss2
from torchdiffeq import odeint_adjoint as odeint
# from torchdiffeq import odeint as odeint2
from VGFM.models import  ODEFunc2


def pretrain(model, df, optimizer, n_epoch=1000, test_interval=100,
    batch_size = 256,
    hold_one_out=False,
    hold_out='random',
    logger=None,
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    save_dir=None,
    relative_mass=None,
    reg = 0.01,
    reg_m = [10,np.inf],
    norm_cost=False,
):
    print('relative_mass',relative_mass)
    time_labels = df['samples'].to_numpy()
    data = df.iloc[:,1:].to_numpy()
    X = [data[time_labels == t] for t in np.unique(time_labels)]
    X_selected = [data[time_labels == t] for t, v in enumerate(np.unique(time_labels)) if v != hold_out]
    t_train = [i for i, v in enumerate(np.unique(time_labels)) if v != hold_out]
    print('t_train:', t_train)
    logger.info('Begin flow and growth matching')
    uot_plans = compute_uot_plans(X_selected, t_train, reg=reg, reg_m=reg_m, draw=False, norm_cost=norm_cost)
    progress_bar = tqdm(range(n_epoch), desc="Begin flow and growth matching...", unit="epoch")
    vloss_list = []
    gloss_list = []
    loss_list = []

    for i in progress_bar:
        optimizer.zero_grad()
        t, xt, ut, gt = get_batch(X_selected, t_train, batch_size, uot_plans, relative_mass)
        vt = model.v_net(t,xt)
        gt_pred = model.g_net(t,xt)
        
        vloss = torch.mean((vt - ut)**2)
        vloss_list.append(vloss.item())
        gloss = torch.mean((gt_pred - gt)**2)
        gloss_list.append(gloss.item())
        loss = vloss + gloss
        loss_list.append(loss.item())
        
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(v.parameters(), max_norm=0.1)
        # torch.nn.utils.clip_grad_norm_(g.parameters(), max_norm=0.1)

        optimizer.step()
        logging.info(f"Epoch {i}: loss={loss.item():.6f}, vloss={vloss.item():.6f}, gloss={gloss.item():.6f}")
        progress_bar.set_postfix({"loss": f"{loss.item():.6f}","vloss": f"{vloss.item():.6f}", "gloss": f"{gloss.item():.6f}"})
    
    return model, vloss_list, gloss_list, loss_list



#%%
def train(
    model, df, groups, optimizer, n_epochs=20, 
    criterion=MMD_loss(),
    use_cuda=False,

    sample_size=(100, ),
    sample_with_replacement=False,

    local_loss=True,
    global_loss=False,

    hold_one_out=False,
    hold_out='random',
    apply_losses_in_time=True,

    device=None,
    logger=None,

    relative_mass=None,
    initial_size=None,
    best_model_path=None,
    stepsize=0.01,
):

    steps = generate_steps(groups)

    # Storage variables for losses
    batch_losses = []
    globe_losses = []
    if hold_one_out and hold_out in groups:
        groups_ho = [g for g in groups if g != hold_out]
        local_losses = {f'{t0}:{t1}':[] for (t0, t1) in generate_steps(groups_ho) if hold_out not in [t0, t1]}
    else:
        local_losses = {f'{t0}:{t1}':[] for (t0, t1) in steps}
        
    # Send model to cuda and specify it as training mode
    if use_cuda:
        model = model.cuda()
    
    model.train()
    model.to(device)

    # Initialize the minimum Otloss with a very high value
    min_ot_loss = float('inf')
    for epoch in tqdm(range(n_epochs)):
        
        # apply local loss
        if local_loss and not global_loss:
            i_mass=1
            lnw0 = torch.log(torch.ones(sample_size[0],1) / (initial_size)).to(device) #torch.Size([n_sample_size, 1])
            data_t0 = sample(df, 0, size=sample_size, replace=sample_with_replacement, to_torch=True, use_cuda=use_cuda)
            data_t0.to(device)
            # print('sample size:',sample_size)
            # for storing the local loss with calling `.item()` so `loss.backward()` can still be used

            batch_loss = []
            if hold_one_out:
                groups = [g for g in groups if g != hold_out] 
                steps = generate_steps(groups)
                print(steps)
            for step_idx, (t0, t1) in enumerate(steps):
                print(f'epoch_{epoch}_{step_idx}:')
                logger.info(f'epoch_{epoch}_{step_idx}:')                      
                optimizer.zero_grad()
                
                #sampling, predicting, and evaluating the loss.
                # sample data
                size1=(df[df['samples']==t1].values.shape[0],)
                data_t1 = sample(df, t1, size=size1, replace=sample_with_replacement, to_torch=True, use_cuda=use_cuda)
                data_t1.to(device)
                time = torch.Tensor([t0, t1]).cuda() if use_cuda else torch.Tensor([t0, t1])
                time.to(device)

                relative_mass_now = relative_mass[i_mass]
                data_t0=data_t0.to(device)
                data_t1=data_t1.to(device)
                initial_state_energy = (data_t0, lnw0)
                t=time.to(device)#(t0,t1)
                t.requires_grad=True
                data_t0.requires_grad=True
                lnw0.requires_grad=True
                
                x_t, lnw_t=odeint(ODEFunc2(model), initial_state_energy, t, options=dict(step_size=stepsize), method='euler')
                lnw_t_last = lnw_t[-1]
                mu = torch.exp(lnw_t_last)
                print(f'relative_mass_pred:{mu.detach().cpu().numpy().sum()},relative_mass_now:{relative_mass_now}')
                print(f'RME:{np.abs(mu.detach().cpu().numpy().sum()-relative_mass_now)/relative_mass_now}')
                logger.info(f'relative_mass_pred:{mu.detach().cpu().numpy().sum()},relative_mass_now:{relative_mass_now}')
                logger.info(f'diff_mass:{mu.detach().cpu().numpy().sum()-relative_mass_now}')
                logger.info(f'RME:{np.abs(mu.detach().cpu().numpy().sum()-relative_mass_now)/relative_mass_now}')
                mu = mu / mu.sum()
                nu = torch.ones(data_t1.shape[0],1)
                nu = nu / nu.sum()
                mu = mu.squeeze(1)
                nu=nu.squeeze(1)
               
                loss_ot = criterion(x_t[-1], data_t1, mu, nu,device=device)
                i_mass=i_mass+1
             
                lnw0=lnw_t_last.detach()
                data_t0=x_t[-1].detach()
            
                print(f'Otloss:{loss_ot.item():.6}')
                logger.info(f'Otloss:{loss_ot.item():.6}')
                
                loss=loss_ot
                print(f'total loss:{loss.item():.6}')
                logger.info(f'total loss:{loss.item():.6}')
        

                # apply local loss as we calculate it
                if apply_losses_in_time and local_loss:
                    loss.backward()
                    optimizer.step()
                # save loss in storage variables 
                local_losses[f'{t0}:{t1}'].append(loss.item())
                batch_loss.append(loss)
               # Detach the loss from the computation graph and get its scalar value
            current_ot_loss = loss_ot.item()
            
            # Check if the current Otloss is the new minimum
            if current_ot_loss < min_ot_loss:
                min_ot_loss = current_ot_loss
                # Save the model's state_dict
                torch.save(model.state_dict(), best_model_path)
                print(f'New minimum otloss found: {min_ot_loss}. Model saved.')
        
        
            # convert the local losses into a tensor of len(steps)
            batch_loss = torch.Tensor(batch_loss).float()
            if use_cuda:
                batch_loss = batch_loss.cuda()
            
            if not apply_losses_in_time:
                batch_loss.backward()
                optimizer.step()

            # store average / sum of local losses for training
            ave_local_loss = torch.mean(batch_loss)          
            batch_losses.append(ave_local_loss.item())
                     
    print_loss = globe_losses if global_loss else batch_losses 
    if logger is None:      
        tqdm.write(f'Train loss: {np.round(np.mean(print_loss), 5)}')
    else:
        logger.info(f'Train loss: {np.round(np.mean(print_loss), 5)}')
    return local_losses, batch_losses, globe_losses
