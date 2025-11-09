import pygame
import numpy as np
import time
from abc import ABC, abstractmethod

# --- CONFIG ---
WIDTH, HEIGHT = 800, 600
FPS_RENDER = 60
DT_PHYSICS = 1/150  # passo fisico (10 ms)
PX_PER_METER = 100.0  # 100 pixel = 1 metro  ### SCALA ###
FRICTION_AIR = 0  # coefficiente di attrito dell'aria
DEBUG_DRAWING = False  # disegna informazioni di debug


# --- CLASSI ---
class SimulationObject(ABC):
    @abstractmethod
    def update(self, dt):
        """Aggiorna lo stato dell'oggetto di dt secondi."""
        raise NotImplementedError

    @abstractmethod
    def draw(self, screen):
        """Disegna l'oggetto sullo schermo di pygame."""
        raise NotImplementedError
    


class Particle(SimulationObject):
    """Particella con massa, posizione e velocità in metri.
    Utilizza l'integrazione Verlet e la separazione delle forze.
    """
    def __init__(self, pos_m, vel_m, radius_m=0.06, color=(255, 200, 40), mass=1.0, static=False, force_fields=[]):
        # Stato fisico (in metri, m/s, kg)
        self.pos = np.array(pos_m, dtype=float)  # posizione [m]
        self.vel = np.array(vel_m, dtype=float)  # velocità [m/s] (per attrito/collisioni)
        
        # Posizione al passo precedente (per integrazione Verlet)
        # Stimata usando la velocità iniziale e il passo di fisica
        self.old_pos = self.pos - self.vel * DT_PHYSICS 
        
        self.radius = float(radius_m)            # raggio [m]
        self.color = color
        self.mass = mass
        # Forza accumulata durante il passo di integrazione
        self.force = np.zeros(2, dtype=float)
        self.static = static
        self.force_fields = force_fields
        self.force_stack = []  # forze temporanee (non usate in questo esempio)

    def get_predicted_pos(self, dt):
        """Calcola la posizione predetta solo in base alle forze applicate"""
        if self.static:
            return self.pos
        # 'self.force' contiene GIA' gravità e attrito (dal passo 1 del loop)
        acc_ext = self.force / self.mass 
        # Calcola la predizione Verlet solo con le forze esterne
        return 2.0 * self.pos - self.old_pos + acc_ext * (dt ** 2)
    
    def get_predicted_vel(self, dt):
        """Calcola la velocità predetta solo in base alle forze applicate"""
        if self.static:
            return np.zeros(2, dtype=float)
        pred_pos = self.get_predicted_pos(dt)
        return (pred_pos - self.old_pos) / dt

    def apply_force(self, f, time=0.0):
        """Aggiunge una forza temporanea alla particella [N]."""
        if not self.static:
            self.force_stack.append((f, time))

    def add_force(self, f):
        """Aggiunge una forza alla particella [N]."""
        self.force += f if not self.static else 0.0

    def apply_external_forces(self, dt):
        """Azzera le forze e applica solo quelle esterne (gravità, attrito)."""
        # Azzera la forza accumulata
        self.force[:] = 0.0
        if self.static:
            return
        
        
        # Applica frizione (usa la velocità calcolata nel passo precedente)
        self.add_force(-FRICTION_AIR * self.vel)
        # Applica campi di forza (es. gravità)
        for ff in self.force_fields:
            self.add_force(ff(self))
        
        for f in self.force_stack:
            self.add_force(f[0])  # applica forze temporanee
        self.force_stack = [ (f, t-dt) for (f,t) in self.force_stack if t > dt ]

    def update(self, dt):
        """Integra la posizione usando la forza accumulata (esterna + interna)."""
        if self.static:
            # Anche se statico, aggiorniamo la vel (per attrito in apply_external_forces)
            self.vel[:] = 0.0
            self.force[:] = 0.0 # Assicurati che la forza sia 0 per il prossimo step
            return
        
        # F = m * a  →  a = F/m
        # 'self.force' ora include forze esterne (da apply_external_forces)
        # E forze delle molle (da SpringConstraint.update)
        acc = self.force / self.mass

        # --- Integrazione Verlet (Position-Based) ---
        new_pos = 2.0 * self.pos - self.old_pos + acc * (dt ** 2)

        # Aggiorna le posizioni per il prossimo ciclo
        self.old_pos = self.pos
        self.pos = new_pos

        # Aggiorna la velocità esplicita (necessaria per attrito e collisioni)
        self.vel = (self.pos - self.old_pos) / dt

    def draw(self, screen):
        """Disegna la particella convertendo le coordinate da metri a pixel."""
        pos_px = self.pos * PX_PER_METER
        radius_px = int(self.radius * PX_PER_METER)
        pygame.draw.circle(screen, self.color, pos_px.astype(int), radius_px)

        if DEBUG_DRAWING and not self.static:
            # mostra posizione e velocità come testo
            text_surf = pygame.font.SysFont(None, 20).render(f"Pos: ({self.pos[0]:.2f}, {self.pos[1]:.2f}) m | Vel: ({self.vel[0]:.2f}, {self.vel[1]:.2f}) m/s", True, (255,255,255))
            screen.blit(text_surf, (pos_px[0] + radius_px + 5, pos_px[1] - radius_px - 5))

class DamperConstraint(SimulationObject):
    """Vincolo viscoso tra due particelle (smorzatore).
    Utilizza la posizione predetta per calcolare la forza (più stabile).
    """
    def __init__(self, p1, p2, beta=10.0, color=(200,200,255)):
        self.p1 = p1
        self.p2 = p2
        self.beta = beta  # costante di smorzamento [N·s/m]
        self.color = color
        

    

    def update(self, dt):
        """Calcola la forza di smorzamento viscosa basata sulle VELOCITÀ PREDETTE."""
        if self.p1.static and self.p2.static:
            return

        # 1 Direzione tra le particelle (basata sulle posizioni predette)
        pred_pos1 = self.p1.get_predicted_pos(dt)
        pred_pos2 = self.p2.get_predicted_pos(dt)
        delta_pos = pred_pos2 - pred_pos1
        dist = np.linalg.norm(delta_pos)
        if dist < 1e-8:
            return
        direction = delta_pos / dist

        # 2 Velocità relative (predette)
        pred_vel1 = self.p1.get_predicted_vel(dt)
        pred_vel2 = self.p2.get_predicted_vel(dt)
        rel_vel = pred_vel2 - pred_vel1

        # 3 Componente di velocità lungo la direzione della connessione
        vel_along_axis = np.dot(rel_vel, direction)

        # 4 Forza di damping (si oppone al moto relativo)
        force = -self.beta * vel_along_axis * direction

        # 5 Applica la forza a ciascuna estremità
        if not self.p1.static:
            self.p1.force += 0.5 * force
        if not self.p2.static:
            self.p2.force -= 0.5 * force

    def draw(self, screen):
        """Disegna lo smorzatore e mostra la differenza di velocità ai suoi capi."""
        # Conversione posizioni in pixel
        p1_px = self.p1.pos * PX_PER_METER
        p2_px = self.p2.pos * PX_PER_METER
        # Disegna la linea dello smorzatore
        pygame.draw.line(screen, self.color, p1_px.astype(int), p2_px.astype(int), 2)
        
        if DEBUG_DRAWING:
            # Calcola il punto medio in pixel
            mid = (p1_px + p2_px) / 2
            # Calcola la differenza di velocità (in m/s)
            vel_diff = np.linalg.norm(self.p2.vel - self.p1.vel)

            # Mostra la differenza di velocità come testo sopra lo smorzatore
            text_surf = pygame.font.SysFont(None, 20).render(f"{vel_diff:.2f} m/s", True, (255,255,0))
            screen.blit(text_surf, mid.astype(int) - np.array([0,10]))

class SpringConstraint(SimulationObject):
    """Vincolo elastico tra due particelle (molla).
    Utilizza la posizione predetta per calcolare la forza (più stabile).
    """
    def __init__(self, p1, p2, k=10.0, rest_length=None, color=(200,200,255)):
        self.p1 = p1
        self.p2 = p2
        self.k = k  # costante elastica [N/m]
        self.color = color
        # Se non specificata, la lunghezza a riposo è la distanza iniziale
        if rest_length is None:
            self.rest_length = np.linalg.norm(p2.pos - p1.pos)
        else:
            self.rest_length = rest_length

    

    def update(self, dt):
        """Calcola la forza elastica basandosi sulla POSIZIONE PREDETTA."""
        
        # 1. Calcola le posizioni predette (dove sarebbero senza le molle)
        pred_pos1 = self.p1.get_predicted_pos(dt)
        pred_pos2 = self.p2.get_predicted_pos(dt)

        # 2. Calcola la forza in base a quelle posizioni predette
        delta = pred_pos2 - pred_pos1
        dist = np.linalg.norm(delta)
        if dist < 1e-8:
            return  # evita divisioni per zero
        
        direction = delta / dist
        # Forza proporzionale alla differenza di lunghezza
        displacement = dist - self.rest_length
        # Dividiamo per 2 per applicare metà forza a ciascuna estremità
        force = (self.k * displacement / 2) * direction
        
        # 3. Applica la forza (che sarà usata da p.integrate() nel passo 3)
        self.p1.force += force
        self.p2.force -= force

    def draw(self, screen):
        """Disegna la molla e mostra la sua deformazione."""
        # Conversione posizioni in pixel
        p1_px = self.p1.pos * PX_PER_METER
        p2_px = self.p2.pos * PX_PER_METER
        # Disegna la linea della molla
        pygame.draw.line(screen, self.color, p1_px.astype(int), p2_px.astype(int), 2)
        
        if DEBUG_DRAWING:
            # Calcola il punto medio in pixel
            mid = (p1_px + p2_px) / 2
            # Calcola la deformazione (in metri)
            displacement = np.linalg.norm(self.p2.pos - self.p1.pos) - self.rest_length

            # Mostra la deformazione come testo sopra la molla
            text_surf = pygame.font.SysFont(None, 20).render(f"{displacement:.2f} m", True, (255,255,0))
            screen.blit(text_surf, mid.astype(int))


class Constraint(SimulationObject):
    """Vincolo rigido (linea) con coefficiente di restituzione."""
    def __init__(self, p1_m, p2_m, restitution=0.9, color=(200,200,200)):
        # Estremi del segmento (in metri)
        self.p1 = np.array(p1_m, dtype=float)
        self.p2 = np.array(p2_m, dtype=float)
        self.restitution = restitution
        self.color = color

    def update(self, dt):
        """I vincoli statici non si aggiornano nel tempo."""
        pass

    def handle_collision(self, particle):
        """Gestisce la collisione tra una particella e il vincolo."""
        seg = self.p2 - self.p1
        seg_len = np.linalg.norm(seg)
        if seg_len == 0:
            return
        seg_dir = seg / seg_len
        normal = np.array([-seg_dir[1], seg_dir[0]])

        rel = particle.pos - self.p1
        dist = np.dot(rel, normal)

        # Se la particella è dall’altro lato, inverti la normale
        if dist < 0:
            normal = -normal
            dist = -dist

        # Se è troppo lontana, nessuna collisione
        if dist > particle.radius:
            return

        # Controlla che il punto proiettato sia dentro al segmento
        proj_len = np.dot(rel, seg_dir)
        if proj_len < 0 or proj_len > seg_len:
            return

        # Calcolo penetrazione e correzione posizione
        penetration = particle.radius - dist
        particle.pos += normal * penetration

        # Calcola la velocità pre-collisione (dal passo di update)
        v_pre = particle.vel
        
        # Riflettiamo la velocità rispetto alla normale (rimbalzo)
        v_n = np.dot(v_pre, normal) * normal
        v_t = v_pre - v_n
        v_post = v_t - v_n * self.restitution # velocità post-collisione
        
        # Aggiorna la velocità della particella
        particle.vel = v_post
        
        # --- MODIFICA CRUCIALE per VERLET ---
        # Aggiorna old_pos per essere coerente
        # con la nuova posizione e la nuova velocità, altrimenti
        # il prossimo step di integrazione "esploderà".
        particle.old_pos = particle.pos - particle.vel * DT_PHYSICS
        # -------------------------------------

    def draw(self, screen):
        """Disegna il vincolo convertendo da metri a pixel."""
        p1_px = self.p1 * PX_PER_METER
        p2_px = self.p2 * PX_PER_METER
        pygame.draw.line(screen, self.color, p1_px.astype(int), p2_px.astype(int), 3)
        if DEBUG_DRAWING:
            # Mostra il vettore normale al centro del vincolo
            mid = (p1_px + p2_px) / 2
            seg = self.p2 - self.p1
            seg_len = np.linalg.norm(seg)
            if seg_len == 0:
                return

            # Direzione e normale del vincolo (in metri)
            seg_dir = seg / seg_len
            normal = np.array([-seg_dir[1], seg_dir[0]])

            # Converti in pixel
            normal_px = normal * 30  # lunghezza della freccia in pixel

            # Calcola punti in pixel
            start = mid
            tip = mid + normal_px

            # Disegna la linea principale della freccia
            pygame.draw.line(screen, (255, 100, 100), start.astype(int), tip.astype(int), 2)

            # Disegna le alette della punta (simmetriche rispetto alla direzione della normale)
            arrow_size = 4 # lunghezza delle alette
            # Per le alette calcoliamo due direzioni oblique rispetto alla normale
            left_wing = tip - normal_px * 0.3 + seg_dir * arrow_size
            right_wing = tip - normal_px * 0.3 - seg_dir * arrow_size

            pygame.draw.line(screen, (255, 100, 100), tip.astype(int), left_wing.astype(int), 2)
            pygame.draw.line(screen, (255, 100, 100), tip.astype(int), right_wing.astype(int), 2)




# --- MAIN LOOP ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Physics Engine 2D — Verlet + Predizione Molle (Stabile)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)


    force_fields = [lambda p: np.array([0.0, 9.81]) * p.mass]
    
    # fai un cubo composto da 9  palline 
    particles = [
        Particle(pos_m=(3.0 + (i%3)*0.60, 1.0 + (i//3)*0.60), vel_m=(0.0, 10.0), radius_m=0.06, color=(255,100,100), mass=0.5, static=False, force_fields=force_fields)
        for i in range(9)
        
    ]
    # connettili tutti insieme
    springs = [
        SpringConstraint(particles[i], particles[j], k=200.0)
        for i in range(len(particles))
        for j in range(i+1, len(particles))
    ]
    dumpers = [
        DamperConstraint(particles[i], particles[j], beta=-0.01)
        for i in range(len(particles))
        for j in range(i+1, len(particles))

    ]

    # Crea vincoli ai bordi (un rettangolo di 7x5 m)
    static_constraints = [
        Constraint((0.5,0.5), (7.5,0.5)),
        Constraint((7.5,0.5), (7.5,5.5),),
        Constraint((7.5,5.5), (0.5,5.5)),
        Constraint((0.5,5.5), (0.5,0.5)),
    ]

    
    
    
    # Lista per il *disegno*
    all_objects_to_draw = springs + dumpers + particles + static_constraints 
    
    center_mass = []  # gruppi di particelle per cui calcolare il centro di massa


    running = True
    accumulator = 0.0
    last_time = time.perf_counter()

    # Loop principale della simulazione
    while running:
        # --- GESTIONE EVENTI ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEWHEEL:
                global PX_PER_METER
                if event.y > 0:
                    PX_PER_METER *= 1.1  # zoom in
                elif event.y < 0:
                    PX_PER_METER /= 1.1  # zoom out
                PX_PER_METER = max(10, min(1000, PX_PER_METER))  # limiti
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    for p in particles:
                        print( f"Particella at pos {p.pos}, vel {p.vel}" )
                        p.apply_force(np.array([3.0, 0.0]), time=1) 
                if event.key == pygame.K_TAB:
                    global DEBUG_DRAWING
                    DEBUG_DRAWING = not DEBUG_DRAWING

        # --- GESTIONE TEMPO ---
        now = time.perf_counter()
        frame_time = now - last_time
        last_time = now
        accumulator += frame_time
        simulation_time = 0.0
        count = 0
        # --- FISICA (Loop Ristrutturato) ---
        while accumulator >= DT_PHYSICS:
            start_simulation_time = time.perf_counter()
            # 1. Azzera le forze e applica forze ESTERNE (gravità, attrito)
            for p in particles:
                p.apply_external_forces(DT_PHYSICS)

            # 2. Calcola e applica forze di interazione (molle)
            #    (usando le posizioni predette)
            for s in springs:
                s.update(DT_PHYSICS) # Questo 'update' applica le forze alle particelle
            
            for d in dumpers:
                d.update(DT_PHYSICS)

            # 3. Integra le posizioni (usa forze esterne + interne)
            for p in particles:
                p.update(DT_PHYSICS)
                
            # 4. Gestisci collisioni (vincoli statici)
            
            for c in static_constraints:
                for p in particles:
                    c.handle_collision(p)
                    
            # --- AGGIORNA I CONTATORI ---
            accumulator -= DT_PHYSICS
            simulation_time = time.perf_counter() - start_simulation_time
            count += 1

        # --- DISPLAY ---
        screen.fill((20, 20, 30))
        
        # Disegna tutti gli oggetti
        for obj in all_objects_to_draw:
            obj.draw(screen)
            pass

        
        # draw_force_fields(screen, force_fields)

        # Mostra il centro di massa del sistema (in pixel)
        for group in center_mass:
            # Calcola la posizione in metri
            cm_pos = sum(p.pos * p.mass for p in group) / sum(p.mass for p in group)
            
            # --- MODIFICA QUI ---
            
            cm_pos_px = cm_pos * PX_PER_METER
            
            cm_radius_px = int(0.04 * PX_PER_METER)
            
            
            pygame.draw.circle(screen, (255,100,100), cm_pos_px.astype(int), max(1, cm_radius_px))
            


        # Mostra testo informativo
        text = font.render(f"Scala: 1 m = {PX_PER_METER:.0f} px", True, (255, 255, 255))
        # mostra il tempo di rendering in ms e il tempo di simulazione in ms
        render_time = clock.get_time()
        render_fps = clock.get_fps()
        text2 = font.render(f"Render Time: {render_time:.2f} ms | FPS: {render_fps:.2f}", True, (255, 255, 255))
        text3 = font.render(f"Physics Step: {DT_PHYSICS*1000:.2f} ms | Sim Time: {simulation_time*1000:.2f} ms | Sim/Frame: {count}", True, (255, 255, 255))
        screen.blit(text2, (10, 30))
        screen.blit(text3, (10, 50))
        screen.blit(text, (10, 10))

        pygame.display.flip()
        clock.tick(FPS_RENDER)

    pygame.quit()

def draw_force_fields(screen, ff_list):
    """Disegna i campi di forza come vettori sullo schermo."""
    step_m = 0.5  # passo in metri
    for x_m in np.arange(0.0, WIDTH / PX_PER_METER, step_m):
        for y_m in np.arange(0.0, HEIGHT / PX_PER_METER, step_m):
            pos = np.array([x_m, y_m])
            p = Particle(pos_m=pos, vel_m=(0.0, 0.0), static=True)
            total_force = np.zeros(2, dtype=float)
            for ff in ff_list:
                total_force += ff(p)
            # Disegna il vettore di forza
            start_px = pos * PX_PER_METER
            end_px = start_px + (total_force * 3)  # scala per visualizzazione
            pygame.draw.line(screen, (100,255,100), start_px.astype(int), end_px.astype(int), 1)
            # Disegna una freccia alla fine
            direction = end_px - start_px
            length = np.linalg.norm(direction)
            if length > 0:
                direction /= length
                perp = np.array([-direction[1], direction[0]]) * 5  # larghezza della freccia
                tip = end_px
                left_wing = tip - direction * 10 + perp
                right_wing = tip - direction * 10 - perp
                pygame.draw.line(screen, (100,255,100), tip.astype(int), left_wing.astype(int), 1)
                pygame.draw.line(screen, (100,255,100), tip.astype(int), right_wing.astype(int), 1)


if __name__ == "__main__":
    main()