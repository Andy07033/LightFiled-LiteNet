"""Procedural two-plane light fields; no benchmark images or labels used."""
import torch


def random_geometry(x, target, step=2, small=False):
    n, _, h, w = x.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing='ij')
    angles = [(v,u) for v in (-step,0,step) for u in (-step,0,step)]
    x, target = x.clone(), target.clone()
    for i in range(n):
        if torch.rand((),device=x.device) >= .25:
            continue
        back = torch.empty((),device=x.device).uniform_(-1.5,.2)
        front = back+torch.empty((),device=x.device).uniform_(.25,1.8)
        colors = torch.empty((2,3,1,1),device=x.device).uniform_(.15,.85)
        if torch.rand((),device=x.device)<.5:
            colors[1] = (colors[0]+torch.empty_like(colors[0]).uniform_(-.12,.12)).clamp(0,1)
        freq = torch.empty((2,2),device=x.device).uniform_(.05,.8)
        phases = torch.empty((2,3,1,1),device=x.device).uniform_(0,6.28)
        centers = torch.rand((8,2),device=x.device)*torch.tensor([w,h],device=x.device)
        radii = torch.empty((8,2),device=x.device).uniform_(1.5,22.)
        rectangles = bool(torch.rand((),device=x.device)<.5)
        if small:
            # Log-uniform sizes expose subpixel/tiny objects; random thin bars
            # cover a different geometry than the default large planar shapes.
            centers = torch.rand((32,2),device=x.device)*torch.tensor([w,h],device=x.device)
            radii = torch.empty((32,2),device=x.device).uniform_(-.43,1.79).exp()
            if rectangles:
                axis = int(torch.randint(2,(),device=x.device))
                radii[:,axis].uniform_(3.,40.)
        views=[]
        for v,u in angles:
            bx,by=xx+u*back,yy+v*back
            fx,fy=xx+u*front,yy+v*front
            dx=(fx[None]-centers[:,0,None,None])/radii[:,0,None,None]
            dy=(fy[None]-centers[:,1,None,None])/radii[:,1,None,None]
            distance=torch.maximum(dx.abs(),dy.abs()) if rectangles else torch.sqrt(dx.square()+dy.square())
            inside=(distance<1).any(0)
            far_color=colors[0]+.06*torch.sin(bx*freq[0,0]+by*freq[0,1]+phases[0])
            near_color=colors[1]+.06*torch.sin(fx*freq[1,0]+fy*freq[1,1]+phases[1])
            views.append(torch.where(inside[None],near_color,far_color).clamp(0,1))
            if u==0 and v==0:
                target[i,0]=torch.where(inside,front,back)
        x[i]=torch.stack(views).reshape(27,h,w)
    return x,target
